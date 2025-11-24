"""
Simulador de Spaced Repetition con UIR/UIC
Basado en el paper de Shiqui sobre Unidades Internacionales de Retención y Comprensión
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Importar módulo de autenticación
import auth

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReviewHistory:
    """Registro de un repaso individual"""
    timestamp: str
    grade: int  # 0=Again, 1=Hard, 2=Good, 3=Easy
    response_time: float  # segundos
    reading_time: float  # segundos hasta mostrar respuesta
    P_recall: float  # probabilidad de recordar estimada
    interval_days: int  # intervalo hasta este repaso
    
@dataclass
class Card:
    """Tarjeta de estudio con metadatos UIR/UIC"""
    id: str
    question: str
    answer: str
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Estado de repaso
    last_review: Optional[str] = None
    next_review: Optional[str] = None
    review_count: int = 0
    
    # Parámetros UIR/UIC
    UIC_local: float = 0.0
    UIR_base: float = 7.0  # días (valor inicial razonable)
    UIR_effective: float = 7.0
    
    # Parámetros Anki clásico
    easiness_factor: float = 2.5
    interval_days: int = 1
    repetition_count: int = 0
    
    # Historial
    history: List[ReviewHistory] = field(default_factory=list)
    
    # Estado y notas (Phase 3)
    status: str = "active"  # active, suspended, archived
    notes: str = ""  # Notas personales
    mnemonics: str = ""  # Técnicas mnemotécnicas

@dataclass
class User:
    """Usuario del sistema"""
    username: str
    password_hash: str  # Hash de la contraseña (no guardar en texto plano)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AppState:
    """Estado global de la aplicación"""
    cards: List[Card] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=lambda: {
        'alpha': 0.2,
        'gamma': 0.15,
        'delta': 0.02,
        'eta': 0.05
    })
    tfidf_matrix: Optional[np.ndarray] = None
    similarity_matrix: Optional[np.ndarray] = None
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

# ============================================================================
# PERSISTENCE
# ============================================================================

DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "state.json")

def ensure_data_dir():
    """Crear directorio de datos si no existe"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "backups"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "auto_backups"), exist_ok=True)

def auto_backup():
    """
    Crear backup automático diario
    Se ejecuta al inicio de la app
    Mantiene últimos 7 días de backups
    """
    try:
        if not os.path.exists(STATE_FILE):
            return
        
        # Asegurar que el directorio existe
        ensure_data_dir()
        
        auto_backup_dir = os.path.join(DATA_DIR, "auto_backups")
        date_str = datetime.now().strftime("%Y%m%d")
        backup_file = os.path.join(auto_backup_dir, f"state_{date_str}.json")
        
        # Solo crear backup si no existe uno de hoy
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy(STATE_FILE, backup_file)
            
            # Limpiar backups antiguos (mantener últimos 7 días)
            all_backups = sorted([
                f for f in os.listdir(auto_backup_dir) 
                if f.startswith("state_") and f.endswith(".json")
            ])
            
            if len(all_backups) > 7:
                for old_backup in all_backups[:-7]:
                    try:
                        os.remove(os.path.join(auto_backup_dir, old_backup))
                    except:
                        pass  # Ignorar errores al eliminar backups antiguos
    except Exception as e:
        # No fallar la app si el backup falla
        # Solo registrar el error silenciosamente
        pass


def save_state(state: AppState):
    """Guardar estado en JSON"""
    ensure_data_dir()
    
    # Backup anterior
    if os.path.exists(STATE_FILE):
        backup_file = os.path.join(DATA_DIR, "backups", 
                                   f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.rename(STATE_FILE, backup_file)
    
    # Convertir a dict serializable
    state_dict = {
        'cards': [asdict(card) for card in state.cards],
        'params': state.params,
        'last_updated': datetime.now().isoformat()
    }
    
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)

def load_state() -> AppState:
    """Cargar estado desde JSON"""
    if not os.path.exists(STATE_FILE):
        return AppState()
    
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        # Reconstruir objetos
        cards = [Card(**card_data) for card_data in state_dict.get('cards', [])]
        # Reconstruir history como objetos ReviewHistory
        for card in cards:
            card.history = [ReviewHistory(**h) if isinstance(h, dict) else h 
                           for h in card.history]
        
        return AppState(
            cards=cards,
            params=state_dict.get('params', AppState().params),
            last_updated=state_dict.get('last_updated', datetime.now().isoformat())
        )
    except Exception as e:
        st.error(f"Error cargando estado: {e}")
        return AppState()

# ============================================================================
# CORE ALGORITHMS - UIR/UIC
# ============================================================================

def compute_uir_from_p(t: float, P: float, epsilon: float = 0.01) -> float:
    """
    Calcula UIR = -t / ln(P) con suavizado para evitar log(0)
    
    Args:
        t: tiempo transcurrido (días)
        P: probabilidad de recordar [0,1]
        epsilon: suavizado para P cercano a 0 o 1
    
    Returns:
        UIR en días
    """
    # Suavizado Laplace
    P_smooth = np.clip(P, epsilon, 1 - epsilon)
    
    if t <= 0:
        return 7.0  # valor por defecto
    
    UIR = -t / np.log(P_smooth)
    return max(1.0, UIR)  # mínimo 1 día

def get_spanish_stop_words() -> List[str]:
    """
    Retorna lista completa de stop words en español
    Incluye palabras interrogativas, conectores y palabras comunes sin valor semántico
    """
    return [
        # Palabras interrogativas (lo más importante para filtrar)
        'qué', 'que', 'cuál', 'cual', 'cuáles', 'cuales', 'cómo', 'como',
        'dónde', 'donde', 'cuándo', 'cuando', 'cuánto', 'cuanto', 'cuántos', 'cuantos',
        'cuánta', 'cuanta', 'cuántas', 'cuantas', 'quién', 'quien', 'quiénes', 'quienes',
        'por', 'qué', 'para', 'porqué', 'porque',
        
        # Verbos copulativos y auxiliares comunes en preguntas
        'es', 'son', 'era', 'eran', 'fue', 'fueron', 'sea', 'sean',
        'está', 'esta', 'están', 'estan', 'estaba', 'estaban',
        'ser', 'estar', 'hay', 'haber', 'sido', 'estado',
        'tiene', 'tienen', 'tenía', 'tenia', 'tenían', 'tenian',
        'hace', 'hacen', 'hizo', 'hicieron',
        
        # Verbos comunes en preguntas
        'significa', 'significan', 'significa', 'sirve', 'sirven',
        'funciona', 'funcionan', 'define', 'definen',
        'representa', 'representan', 'implica', 'implican',
        
        # Artículos
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        
        # Preposiciones
        'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'en',
        'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según',
        'sin', 'sobre', 'tras', 'durante', 'versus', 'vía',
        
        # Conjunciones
        'y', 'e', 'o', 'u', 'pero', 'sino', 'aunque', 'si', 'ni',
        'que', 'porque', 'pues', 'ya', 'sea', 'bien', 'así',
        
        # Pronombres
        'yo', 'tú', 'tu', 'él', 'el', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas',
        'me', 'te', 'se', 'nos', 'os', 'le', 'les', 'lo', 'la', 'los', 'las',
        'mi', 'mis', 'su', 'sus', 'nuestro', 'nuestra', 'vuestro', 'vuestra',
        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
        'aquel', 'aquella', 'aquellos', 'aquellas', 'esto', 'eso', 'aquello',
        
        # Adverbios comunes
        'muy', 'más', 'mas', 'menos', 'poco', 'mucho', 'bastante', 'demasiado',
        'tan', 'tanto', 'también', 'tampoco', 'sí', 'si', 'no', 'nunca', 'siempre',
        'jamás', 'jamas', 'apenas', 'solo', 'sólo', 'solamente',
        'aquí', 'aqui', 'ahí', 'ahi', 'allí', 'alli', 'acá', 'aca', 'allá', 'alla',
        'hoy', 'ayer', 'mañana', 'ahora', 'luego', 'después', 'despues', 'antes',
        
        # Otros
        'otro', 'otra', 'otros', 'otras', 'mismo', 'misma', 'mismos', 'mismas',
        'tal', 'tales', 'todo', 'toda', 'todos', 'todas', 'algún', 'algun',
        'alguno', 'alguna', 'algunos', 'algunas', 'ningún', 'ningun', 'ninguno', 'ninguna',
        'cada', 'varios', 'varias', 'ambos', 'ambas', 'cualquier', 'cualesquiera',
        
        # Palabras de relleno
        'cosa', 'cosas', 'algo', 'nada', 'alguien', 'nadie', 'vez', 'veces'
    ]


@st.cache_data
def compute_tfidf(cards_data: Tuple[Tuple[str, str, str], ...]) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """
    Construye matriz TF-IDF de preguntas + respuestas
    Filtra stop words en español (palabras interrogativas, conectores, etc.)
    para enfocarse en palabras núcleo con valor semántico
    
    Args:
        cards_data: Tupla de tuplas (id, question, answer) para hashing
    
    Returns:
        (matriz TF-IDF, vectorizer) o (None, None) si no hay suficientes tarjetas
    """
    if len(cards_data) < 2:
        return None, None
    
    # Reconstruir documentos desde cards_data
    documents = [f"{q} {a}" for _, q, a in cards_data]
    
    try:
        # Obtener stop words personalizadas
        custom_stop_words = get_spanish_stop_words()
        
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words=custom_stop_words,  # Filtrar palabras sin valor semántico
            ngram_range=(1, 2),  # Unigramas y bigramas
            lowercase=True,  # Normalizar a minúsculas
            strip_accents='unicode',  # Normalizar acentos para mejor matching
            token_pattern=r'(?u)\b\w\w+\b'  # Palabras de 2+ caracteres
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        return tfidf_matrix.toarray(), vectorizer
    except Exception as e:
        st.warning(f"Error calculando TF-IDF: {e}")
        return None, None

def compute_tfidf_from_cards(cards: List[Card]) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """
    Wrapper para compute_tfidf que convierte List[Card] a formato cacheable
    """
    # Convertir cards a tupla de tuplas para que sea hashable
    cards_data = tuple((c.id, c.question, c.answer) for c in cards)
    return compute_tfidf(cards_data)



def compute_similarity_matrix(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula matriz de similitud coseno rectificada [0,1]
    Diagonal = 0 (no auto-similitud)
    """
    W = cosine_similarity(tfidf_matrix)
    W = np.clip(W, 0, 1)  # rectificar a [0,1]
    np.fill_diagonal(W, 0)  # diagonal a 0
    return W

def compute_UIC_global(W: np.ndarray) -> float:
    """
    UIC global = sum(w_ij) / (n*(n-1))
    """
    n = W.shape[0]
    if n < 2:
        return 0.0
    
    total_similarity = np.sum(W)
    return total_similarity / (n * (n - 1))

def compute_UIC_local(W: np.ndarray, card_idx: int, k: int = 5) -> float:
    """
    UIC local = promedio de similitud entre k vecinos más cercanos
    
    Args:
        W: matriz de similitud
        card_idx: índice de la tarjeta
        k: número de vecinos a considerar
    """
    n = W.shape[0]
    if n < 2:
        return 0.0
    
    # Obtener similitudes con otras tarjetas
    similarities = W[card_idx, :]
    
    # Top k vecinos (excluyendo la propia tarjeta)
    k_actual = min(k, n - 1)
    if k_actual == 0:
        return 0.0
    
    top_k_indices = np.argsort(similarities)[-k_actual:]
    
    # UIC local = promedio de similitud entre vecinos
    if len(top_k_indices) < 2:
        return np.mean(similarities[top_k_indices]) if len(top_k_indices) > 0 else 0.0
    
    # Similitud promedio entre pares de vecinos
    neighbor_similarities = []
    for i in range(len(top_k_indices)):
        for j in range(i + 1, len(top_k_indices)):
            neighbor_similarities.append(W[top_k_indices[i], top_k_indices[j]])
    
    return np.mean(neighbor_similarities) if neighbor_similarities else 0.0

def update_on_review(card: Card, grade: int, response_time: float, 
                    reading_time: float, params: Dict[str, float]):
    """
    Actualiza UIC_local, UIR_base, UIR_eff según fórmulas del paper
    
    Args:
        card: tarjeta a actualizar
        grade: 0=Again, 1=Hard, 2=Good, 3=Easy
        response_time: tiempo de respuesta en segundos
        reading_time: tiempo de lectura en segundos
        params: parámetros del modelo (alpha, gamma, delta, eta)
    """
    # Mapear grade a probabilidad de recordar
    grade_to_p = {0: 0.0, 1: 0.4, 2: 0.7, 3: 0.95}
    p_t = grade_to_p.get(grade, 0.5)
    
    # Actualizar UIC (ecuación discreta)
    gamma = params['gamma']
    delta = params['delta']
    
    UIC_old = card.UIC_local
    UIC_increment = gamma * p_t * (1 - UIC_old)
    UIC_decrement = delta * (1 - p_t) * UIC_old
    card.UIC_local = np.clip(UIC_old + UIC_increment - UIC_decrement, 0, 1)
    
    # Actualizar UIR_base
    eta = params['eta']
    card.UIR_base = card.UIR_base + eta * p_t * card.UIC_local
    card.UIR_base = max(1.0, card.UIR_base)
    
    # Calcular UIR_effective
    alpha = params['alpha']
    card.UIR_effective = card.UIR_base * (1 + alpha * card.UIC_local)
    
    # Registrar en historial
    interval_days = 0
    if card.last_review:
        last_dt = datetime.fromisoformat(card.last_review)
        interval_days = (datetime.now() - last_dt).days
    
    review = ReviewHistory(
        timestamp=datetime.now().isoformat(),
        grade=grade,
        response_time=response_time,
        reading_time=reading_time,
        P_recall=p_t,
        interval_days=interval_days
    )
    card.history.append(review)
    
    # Actualizar metadatos
    card.last_review = datetime.now().isoformat()
    card.review_count += 1

# ============================================================================
# SCHEDULING ALGORITHMS
# ============================================================================

def compute_success_rate(card: Card) -> float:
    """
    Calcula tasa de éxito reciente (últimos 5 repasos)
    
    Returns:
        Float entre 0 y 1 (0.5 si no hay historial)
    """
    if not card.history:
        return 0.5  # Neutral para tarjetas nuevas
    
    recent = card.history[-5:]  # Últimos 5 repasos
    successes = 0
    for r in recent:
        # Manejar tanto dict como ReviewHistory object
        grade = r.grade if hasattr(r, 'grade') else r.get('grade', 0)
        if grade >= 2:  # Good o Easy
            successes += 1
    
    return successes / len(recent)

def compute_anki_interval_pure(n: int, EF: float, I_prev: int, grade: int) -> Tuple[int, float, int]:
    """
    Calcula intervalo Anki sin modificar la tarjeta (función pura)
    
    Returns:
        (nuevo_intervalo, nuevo_EF, nuevo_n)
    """
    if grade == 0:  # Again
        return 1, max(1.3, EF - 0.2), 0
    elif grade == 1:  # Hard
        return max(1, round(I_prev * 1.2)), max(1.3, EF - 0.15), n + 1
    elif grade == 2:  # Good
        if n == 0:
            return 1, EF, n + 1
        elif n == 1:
            return 6, EF, n + 1
        else:
            return round(I_prev * EF), EF, n + 1
    else:  # Easy (grade == 3)
        if n == 0:
            return 4, EF + 0.1, n + 1
        else:
            return round(I_prev * EF * 1.3), EF + 0.1, n + 1

def anki_classic_schedule(card: Card, grade: int) -> int:
    """
    Algoritmo Anki clásico (SM-2 simplificado)
    Modifica la tarjeta in-place
    
    Returns:
        Próximo intervalo en días
    """
    I_new, EF_new, n_new = compute_anki_interval_pure(
        card.repetition_count,
        card.easiness_factor,
        card.interval_days,
        grade
    )
    
    # Actualizar tarjeta
    card.interval_days = I_new
    card.easiness_factor = EF_new
    card.repetition_count = n_new
    
    return I_new

def compute_uir_modulation_factor(card: Card, grade: int, params: Dict[str, float]) -> float:
    """
    Calcula factor de modulación basado en UIR/UIC
    
    Factor combina:
    - Progreso de retención (UIR_eff / UIR_inicial)
    - Refuerzo semántico (UIC_local)
    - Historial de éxito
    - Dificultad percibida (grade)
    
    Returns:
        Factor entre 0.5 y 2.5
    """
    UIR_INICIAL = 7.0  # UIR de referencia inicial
    
    # 1. Ratio UIR (progreso de retención)
    UIR_ratio = card.UIR_effective / UIR_INICIAL
    
    # 2. Factor UIC (refuerzo semántico)
    # Tarjetas conectadas → intervalos más largos
    UIC_factor = 1 + params['alpha'] * card.UIC_local
    
    # 3. Factor de éxito (historial reciente)
    success_rate = compute_success_rate(card)
    success_factor = 0.7 + 0.6 * success_rate  # Rango [0.7, 1.3]
    
    # 4. Factor de dificultad percibida
    grade_factors = {
        0: 0.5,   # Again: acortar mucho
        1: 0.8,   # Hard: acortar un poco
        2: 1.0,   # Good: neutral
        3: 1.3    # Easy: alargar
    }
    grade_factor = grade_factors.get(grade, 1.0)
    
    # Combinar todos los factores
    total_factor = UIR_ratio * UIC_factor * success_factor * grade_factor
    
    # Limitar rango para evitar extremos
    return np.clip(total_factor, 0.5, 2.5)

def anki_uir_adapted_schedule(card: Card, grade: int, params: Dict[str, float]) -> int:
    """
    Algoritmo híbrido Anki+UIR mejorado
    
    Combina:
    - Intervalo base de Anki (experiencia acumulada)
    - Factor de modulación UIR/UIC (retención individual + contexto semántico)
    
    Returns:
        Próximo intervalo en días
    """
    # 1. Calcular intervalo Anki (sin modificar card)
    I_anki, _, _ = compute_anki_interval_pure(
        card.repetition_count,
        card.easiness_factor,
        card.interval_days,
        grade
    )
    
    # 2. Calcular factor de modulación UIR
    UIR_factor = compute_uir_modulation_factor(card, grade, params)
    
    # 3. Aplicar modulación
    I_final = round(I_anki * UIR_factor)
    
    return max(1, I_final)

def predict_intervals_for_all_grades(card: Card, params: Dict[str, float]) -> Dict[str, Dict[int, int]]:
    """
    Predice intervalos para todas las opciones de calificación
    Útil para mostrar en UI antes de que el usuario elija
    
    Returns:
        {
            'anki_classic': {0: días, 1: días, 2: días, 3: días},
            'anki_uir': {0: días, 1: días, 2: días, 3: días}
        }
    """
    predictions = {
        'anki_classic': {},
        'anki_uir': {}
    }
    
    for grade in [0, 1, 2, 3]:
        # Anki clásico (función pura, no modifica card)
        I_anki, _, _ = compute_anki_interval_pure(
            card.repetition_count,
            card.easiness_factor,
            card.interval_days,
            grade
        )
        predictions['anki_classic'][grade] = I_anki
        
        # Anki+UIR
        UIR_factor = compute_uir_modulation_factor(card, grade, params)
        I_uir = round(I_anki * UIR_factor)
        predictions['anki_uir'][grade] = max(1, I_uir)
    
    return predictions

def compute_next_review_date(card: Card, interval_days: int) -> str:
    """Calcula fecha de próximo repaso"""
    next_date = datetime.now() + timedelta(days=interval_days)
    return next_date.isoformat()

def compute_streak(cards: List[Card]) -> int:
    """
    Calcula la racha actual de días consecutivos con repasos
    
    Returns:
        Número de días consecutivos con al menos un repaso
    """
    if not cards:
        return 0
    
    # Obtener todas las fechas únicas de repasos
    review_dates = set()
    for card in cards:
        for review in card.history:
            timestamp = review.timestamp if hasattr(review, 'timestamp') else review.get('timestamp', '')
            if timestamp:
                try:
                    date = datetime.fromisoformat(timestamp).date()
                    review_dates.add(date)
                except:
                    pass
    
    if not review_dates:
        return 0
    
    # Ordenar fechas de más reciente a más antigua
    sorted_dates = sorted(review_dates, reverse=True)
    
    # Calcular racha
    today = datetime.now().date()
    streak = 0
    
    # Verificar si hay repaso hoy o ayer (para no romper racha)
    if sorted_dates[0] not in [today, today - timedelta(days=1)]:
        return 0
    
    # Contar días consecutivos
    expected_date = sorted_dates[0]
    for date in sorted_dates:
        if date == expected_date:
            streak += 1
            expected_date = date - timedelta(days=1)
        else:
            break
    
    return streak

# ============================================================================
# STREAMLIT APP INITIALIZATION
# ============================================================================

# Ejecutar backup automático al inicio
auto_backup()

# ============================================================================
# AUTHENTICATION CHECK
# ============================================================================

# Inicializar session state para autenticación
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

# Inicializar dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Si no está autenticado, mostrar página de login
if not st.session_state.authenticated:
    auth.show_auth_page()
    st.stop()

# ============================================================================
# MAIN APP (Solo si está autenticado)
# ============================================================================

# Cargar estado específico del usuario
username = st.session_state.username
STATE_FILE = auth.get_user_state_file(username)

# Inicializar session state
if 'state' not in st.session_state:
    st.session_state.state = load_state()

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

if 'review_session' not in st.session_state:
    st.session_state.review_session = {
        'active': False,
        'current_card_idx': 0,
        'cards_to_review': [],
        'show_answer': False,
        'start_time': None,
        'show_time': None
    }

state = st.session_state.state

# ============================================================================
# NAVIGATION
# ============================================================================

st.sidebar.title("🧠 Simulador UIR/UIC")
st.sidebar.markdown("---")

pages = [
    "Dashboard",
    "Crear/Importar Tarjetas",
    "Sesión de Repaso",
    "📊 Analytics",
    "Grafo Semántico",
    "Comparador de Algoritmos",
    "Simulación",
    "Calibración",
    "Export/Import"
]

st.session_state.current_page = st.sidebar.radio("Navegación", pages, 
                                                  index=pages.index(st.session_state.current_page))

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Tarjetas totales:** {len(state.cards)}")
st.sidebar.markdown(f"**UIC global:** {compute_UIC_global(state.similarity_matrix) if state.similarity_matrix is not None else 0:.3f}")

# Mostrar racha
if state.cards:
    streak = compute_streak(state.cards)
    if streak > 0:
        st.sidebar.markdown(f"🔥 **Racha:** {streak} días")
        if streak >= 7:
            st.sidebar.success("¡Semana completa!")
        if streak >= 30:
            st.sidebar.success("🏆 ¡Mes completo!")

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def page_dashboard():
    """Dashboard principal con métricas y resumen"""
    st.title("📊 Dashboard")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tarjetas", len(state.cards))
    
    with col2:
        uic_global = compute_UIC_global(state.similarity_matrix) if state.similarity_matrix is not None else 0
        st.metric("UIC Global", f"{uic_global:.3f}")
    
    with col3:
        avg_uir = np.mean([c.UIR_effective for c in state.cards]) if state.cards else 0
        st.metric("UIR Promedio", f"{avg_uir:.1f} días")
    
    with col4:
        # Tarjetas pendientes hoy
        today = datetime.now()
        pending = sum(1 for c in state.cards 
                     if c.next_review and datetime.fromisoformat(c.next_review) <= today)
        st.metric("Pendientes Hoy", pending)
    
    st.markdown("---")
    
    # Botón de inicio rápido
    if st.button("🚀 Empezar Sesión de Repaso", type="primary", use_container_width=True):
        st.session_state.current_page = "Sesión de Repaso"
        st.rerun()
    
    # Gráficas mejoradas
    if state.cards:
        # Tabs para organizar visualizaciones
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Actividad", "📊 Distribuciones", "🔄 Retención", "📅 Timeline"])
        
        with tab1:
            st.subheader("Actividad de Repasos")
            
            # Recopilar historial de todos los repasos
            all_reviews = []
            for card in state.cards:
                for review in card.history:
                    timestamp = review.timestamp if hasattr(review, 'timestamp') else review.get('timestamp', '')
                    grade = review.grade if hasattr(review, 'grade') else review.get('grade', 0)
                    if timestamp:
                        all_reviews.append({
                            'timestamp': datetime.fromisoformat(timestamp),
                            'grade': grade
                        })
            
            if all_reviews:
                df_reviews = pd.DataFrame(all_reviews)
                df_reviews['date'] = df_reviews['timestamp'].dt.date
                
                # Gráfica de barras por día
                daily_reviews = df_reviews.groupby('date').size().reset_index(name='count')
                fig1 = px.bar(daily_reviews, x='date', y='count', 
                            title="Repasos por Día",
                            labels={'date': 'Fecha', 'count': 'Número de Repasos'})
                st.plotly_chart(fig1, use_container_width=True)
                
                # Calendario de calor (heatmap de actividad)
                df_reviews['day_of_week'] = df_reviews['timestamp'].dt.day_name()
                df_reviews['hour'] = df_reviews['timestamp'].dt.hour
                heatmap_data = df_reviews.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
                
                # Ordenar días de la semana
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
                heatmap_pivot = heatmap_pivot.reindex(day_order)
                
                fig_heat = px.imshow(heatmap_pivot,
                                    labels=dict(x="Hora del Día", y="Día de la Semana", color="Repasos"),
                                    title="Patrón de Actividad (Heatmap)",
                                    color_continuous_scale="Blues")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("No hay repasos registrados aún.")
        
        with tab2:
            st.subheader("Distribuciones")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Distribución de UIC
                uic_values = [c.UIC_local for c in state.cards]
                fig_uic = px.histogram(x=uic_values, nbins=20,
                                      title="Distribución de UIC Local",
                                      labels={'x': 'UIC Local', 'y': 'Frecuencia'})
                st.plotly_chart(fig_uic, use_container_width=True)
            
            with col_b:
                # Distribución de UIR
                uir_values = [c.UIR_effective for c in state.cards]
                fig_uir = px.histogram(x=uir_values, nbins=20,
                                      title="Distribución de UIR Efectivo",
                                      labels={'x': 'UIR Efectivo (días)', 'y': 'Frecuencia'})
                st.plotly_chart(fig_uir, use_container_width=True)
            
            # Distribución de intervalos
            intervals = []
            for c in state.cards:
                if c.next_review:
                    try:
                        next_dt = datetime.fromisoformat(c.next_review)
                        days_until = (next_dt - datetime.now()).days
                        intervals.append(max(0, days_until))
                    except:
                        pass
            
            if intervals:
                fig_int = px.histogram(x=intervals, nbins=30,
                                      title="Distribución de Próximos Repasos",
                                      labels={'x': 'Días hasta próximo repaso', 'y': 'Frecuencia'})
                st.plotly_chart(fig_int, use_container_width=True)
        
        with tab3:
            st.subheader("Curva de Retención Promedio")
            
            # Calcular curva P(t) = exp(-t/UIR_eff) promedio
            avg_uir_eff = np.mean([c.UIR_effective for c in state.cards]) if state.cards else 7.0
            
            t_values = np.linspace(0, 90, 100)
            p_values = np.exp(-t_values / avg_uir_eff)
            
            fig_retention = go.Figure()
            fig_retention.add_trace(go.Scatter(x=t_values, y=p_values,
                                              mode='lines',
                                              name=f'P(t) = exp(-t/{avg_uir_eff:.1f})',
                                              line=dict(color='blue', width=3)))
            
            # Línea de referencia (37% en t=UIR)
            fig_retention.add_hline(y=0.37, line_dash="dash", line_color="red",
                                   annotation_text="37% (1/e)")
            fig_retention.add_vline(x=avg_uir_eff, line_dash="dash", line_color="red",
                                   annotation_text=f"UIR={avg_uir_eff:.1f}d")
            
            fig_retention.update_layout(
                title="Curva de Retención Promedio",
                xaxis_title="Tiempo (días)",
                yaxis_title="Probabilidad de Recordar",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_retention, use_container_width=True)
            
            st.caption(f"📊 UIR promedio: {avg_uir_eff:.1f} días - Probabilidad cae a 37% después de {avg_uir_eff:.1f} días")
        
        with tab4:
            st.subheader("Evolución Temporal")
            
            # Timeline de UIR y UIC
            timeline_data = []
            for card in state.cards:
                for i, review in enumerate(card.history):
                    timestamp = review.timestamp if hasattr(review, 'timestamp') else review.get('timestamp', '')
                    if timestamp:
                        timeline_data.append({
                            'timestamp': datetime.fromisoformat(timestamp),
                            'UIR_base': card.UIR_base,  # Valor actual (simplificado)
                            'UIC_local': card.UIC_local
                        })
            
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                df_timeline = df_timeline.sort_values('timestamp')
                
                # Agrupar por día y promediar
                df_timeline['date'] = df_timeline['timestamp'].dt.date
                daily_avg = df_timeline.groupby('date').agg({
                    'UIR_base': 'mean',
                    'UIC_local': 'mean'
                }).reset_index()
                
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['UIR_base'],
                                                 mode='lines+markers', name='UIR Base Promedio',
                                                 line=dict(color='blue')))
                fig_timeline.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['UIC_local']*10,
                                                 mode='lines+markers', name='UIC Local Promedio (×10)',
                                                 line=dict(color='green')))
                
                fig_timeline.update_layout(
                    title="Evolución de UIR y UIC en el Tiempo",
                    xaxis_title="Fecha",
                    yaxis_title="Valor",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No hay suficiente historial para mostrar evolución temporal.")
    else:
        st.info("👋 ¡Bienvenido! Comienza importando tarjetas en la página 'Crear/Importar Tarjetas'.")

def page_import():
    """Página para crear e importar tarjetas"""
    st.title("📥 Crear / Importar Tarjetas")
    
    tab1, tab2, tab3 = st.tabs(["✍️ Texto", "📄 CSV", "🔗 RemNote"])
    
    with tab1:
        st.subheader("Crear tarjetas desde texto")
        st.markdown("Formato: `pregunta == respuesta` (una por línea)")
        
        text_input = st.text_area("Ingresa tus tarjetas:", height=200,
                                  placeholder="¿Qué es Python? == Un lenguaje de programación\n¿Qué es Streamlit? == Framework para apps de datos")
        
        tags_input = st.text_input("Etiquetas (separadas por comas):", placeholder="python, programación")
    with tab2:
        st.subheader("Importar desde CSV")
        st.markdown("El CSV debe tener columnas: `question,answer` o `front,back` o `item,note`")
        
        uploaded_file = st.file_uploader("Selecciona archivo CSV", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Vista previa:", df.head())
                
                # Detectar columnas
                q_col = None
                a_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in ['question', 'pregunta', 'front', 'item']:
                        q_col = col
                    if col_lower in ['answer', 'respuesta', 'back', 'note']:
                        a_col = col
                
                if q_col and a_col:
                    st.success(f"✅ Detectadas columnas: `{q_col}` y `{a_col}`")
                    
                    tags_csv = st.text_input("Etiquetas para todas las tarjetas:", key="csv_tags")
                    
                    if st.button("Importar Tarjetas"):
                        tags = [t.strip() for t in tags_csv.split(',') if t.strip()]
                        created_count = 0
                        
                        for _, row in df.iterrows():
                            question = str(row[q_col]).strip()
                            answer = str(row[a_col]).strip()
                            
                            if question and answer:
                                card = Card(
                                    id=f"card_{len(state.cards)}_{int(time.time())}_{created_count}",
                                    question=question,
                                    answer=answer,
                                    tags=tags
                                )
                                state.cards.append(card)
                                created_count += 1
                        
                        save_state(state)
                        st.success(f"✅ {created_count} tarjetas importadas!")
                        st.rerun()
                else:
                    st.error("No se encontraron columnas válidas. Asegúrate de tener 'question' y 'answer'.")
            except Exception as e:
                st.error(f"Error leyendo CSV: {e}")
    
    
    with tab3:
        st.subheader("Importar desde RemNote (Markdown)")
        st.markdown("RemNote exporta en formato Markdown. Formato: `- Pregunta >>> Respuesta`")
        st.code("""Ejemplo:
- ¿Cuáles son los propósitos de la ciencia? >>>
    - Quitar lo superficial y entrar a la esencia
- ¿Qué es Python? >>> Un lenguaje de programación
        """, language="markdown")
        
        markdown_input = st.text_area("Pega tu export de RemNote (Markdown):", height=300,
                                      placeholder="- ¿Pregunta 1? >>>\n    - Respuesta 1\n- ¿Pregunta 2? >>> Respuesta 2")
        
        tags_md = st.text_input("Etiquetas (separadas por comas):", placeholder="remnote, estudio", key="md_tags")
        
        if st.button("Importar desde Markdown"):
            if markdown_input.strip():
                created_count = 0
                errors = []
                tags = [t.strip() for t in tags_md.split(',') if t.strip()]
                
                # Parser para formato RemNote Markdown
                lines = markdown_input.strip().split('\n')
                current_question = None
                current_answer_lines = []
                
                for i, line in enumerate(lines):
                    line_stripped = line.strip()
                    
                    # Detectar pregunta (línea que contiene >>>)
                    if '>>>' in line:
                        # Guardar tarjeta anterior si existe
                        if current_question and current_answer_lines:
                            answer = ' '.join(current_answer_lines).strip()
                            if answer:
                                card = Card(
                                    id=f"card_{len(state.cards)}_{int(time.time())}_{created_count}",
                                    question=current_question,
                                    answer=answer,
                                    tags=tags
                                )
                                state.cards.append(card)
                                created_count += 1
                            current_answer_lines = []
                        
                        # Procesar nueva pregunta
                        parts = line.split('>>>', 1)
                        question_part = parts[0].strip()
                        
                        # Limpiar bullets y guiones
                        question_part = question_part.lstrip('-').lstrip('*').strip()
                        current_question = question_part
                        
                        # Si hay respuesta en la misma línea
                        if len(parts) > 1 and parts[1].strip():
                            answer_part = parts[1].strip()
                            answer_part = answer_part.lstrip('-').lstrip('*').strip()
                            current_answer_lines.append(answer_part)
                    
                    # Líneas de respuesta (indentadas o con bullets)
                    elif current_question and line_stripped:
                        # Limpiar indentación y bullets
                        answer_line = line_stripped.lstrip('-').lstrip('*').strip()
                        if answer_line:
                            current_answer_lines.append(answer_line)
                    
                    # Línea vacía o nueva sección sin >>>
                    elif not line_stripped and current_question:
                        # Guardar tarjeta actual
                        if current_answer_lines:
                            answer = ' '.join(current_answer_lines).strip()
                            if answer:
                                card = Card(
                                    id=f"card_{len(state.cards)}_{int(time.time())}_{created_count}",
                                    question=current_question,
                                    answer=answer,
                                    tags=tags
                                )
                                state.cards.append(card)
                                created_count += 1
                        current_question = None
                        current_answer_lines = []
                
                # Guardar última tarjeta si existe
                if current_question and current_answer_lines:
                    answer = ' '.join(current_answer_lines).strip()
                    if answer:
                        card = Card(
                            id=f"card_{len(state.cards)}_{int(time.time())}_{created_count}",
                            question=current_question,
                            answer=answer,
                            tags=tags
                        )
                        state.cards.append(card)
                        created_count += 1
                
                if created_count > 0:
                    save_state(state)
                    st.success(f"✅ {created_count} tarjetas importadas desde RemNote!")
                    
                    # Mostrar preview
                    with st.expander("Ver tarjetas importadas"):
                        for card in state.cards[-created_count:]:
                            st.markdown(f"**Q:** {card.question}")
                            st.markdown(f"**A:** {card.answer}")
                            st.markdown("---")
                    
                    st.rerun()
                else:
                    st.warning("⚠️ No se encontraron tarjetas válidas. Asegúrate de usar el formato: `- Pregunta >>> Respuesta`")
                    
                if errors:
                    with st.expander("⚠️ Errores encontrados"):
                        for error in errors:
                            st.write(error)


def page_review_session():
    """Sesión interactiva de repaso"""
    st.title("🎯 Sesión de Repaso")
    
    session = st.session_state.review_session
    
    if not session['active']:
        # Seleccionar tarjetas para repasar
        st.subheader("Iniciar Sesión")
        
        if not state.cards:
            st.warning("No hay tarjetas disponibles. Crea algunas primero.")
            return
        
        # Selector de modo de repaso
        st.markdown("### Modo de Repaso")
        review_mode = st.selectbox(
            "Elige cómo quieres repasar:",
            [
                "Pendientes (por fecha)",
                "Aleatorio",
                "Por tag específico",
                "Tarjetas difíciles (éxito < 50%)",
                "Solo tarjetas nuevas",
                "Repaso espaciado óptimo (por UIR)"
            ]
        )
        
        # Filtrar tarjetas según el modo
        today = datetime.now()
        cards_to_review_indices = []
        
        if review_mode == "Pendientes (por fecha)":
            cards_to_review_indices = [i for i, c in enumerate(state.cards)
                                      if not c.next_review or datetime.fromisoformat(c.next_review) <= today]
        
        elif review_mode == "Aleatorio":
            cards_to_review_indices = list(range(len(state.cards)))
            import random
            random.shuffle(cards_to_review_indices)
        
        elif review_mode == "Por tag específico":
            # Obtener todos los tags únicos
            all_tags = set()
            for card in state.cards:
                all_tags.update(card.tags)
            
            if all_tags:
                selected_tag = st.selectbox("Selecciona tag:", sorted(all_tags))
                cards_to_review_indices = [i for i, c in enumerate(state.cards)
                                          if selected_tag in c.tags]
            else:
                st.warning("No hay tags disponibles.")
                return
        
        elif review_mode == "Tarjetas difíciles (éxito < 50%)":
            for i, card in enumerate(state.cards):
                if card.history:
                    # Calcular tasa de éxito
                    recent = card.history[-5:]
                    successes = sum(1 for r in recent 
                                  if (r.grade if hasattr(r, 'grade') else r.get('grade', 0)) >= 2)
                    success_rate = successes / len(recent)
                    if success_rate < 0.5:
                        cards_to_review_indices.append(i)
        
        elif review_mode == "Solo tarjetas nuevas":
            cards_to_review_indices = [i for i, c in enumerate(state.cards)
                                      if c.review_count == 0]
        
        elif review_mode == "Repaso espaciado óptimo (por UIR)":
            # Ordenar por UIR efectivo (menor primero = más urgente)
            card_uir_pairs = [(i, c.UIR_effective) for i, c in enumerate(state.cards)]
            card_uir_pairs.sort(key=lambda x: x[1])
            cards_to_review_indices = [i for i, _ in card_uir_pairs]
        
        # Mostrar información
        st.write(f"**Tarjetas en este modo:** {len(cards_to_review_indices)}")
        st.write(f"**Total tarjetas:** {len(state.cards)}")
        
        # Botones de inicio
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Iniciar Sesión", type="primary", disabled=len(cards_to_review_indices)==0):
                session['active'] = True
                session['cards_to_review'] = cards_to_review_indices
                session['current_card_idx'] = 0
                session['show_answer'] = False
                session['start_time'] = time.time()
                st.rerun()
        
        with col2:
            if st.button("📋 Repasar Todas"):
                session['active'] = True
                session['cards_to_review'] = list(range(len(state.cards)))
                session['current_card_idx'] = 0
                session['show_answer'] = False
                session['start_time'] = time.time()
                st.rerun()
    
    else:
        # Sesión activa
        cards_to_review = session['cards_to_review']
        current_idx = session['current_card_idx']
        
        if current_idx >= len(cards_to_review):
            st.success("🎉 ¡Sesión completada!")
            st.balloons()
            
            if st.button("Volver al Dashboard"):
                session['active'] = False
                st.session_state.current_page = "Dashboard"
                st.rerun()
            return
        
        card_idx = cards_to_review[current_idx]
        card = state.cards[card_idx]
        
        # Mostrar progreso
        st.progress((current_idx + 1) / len(cards_to_review))
        st.caption(f"Tarjeta {current_idx + 1} de {len(cards_to_review)}")
        
        # Mostrar pregunta
        st.markdown(f"### {card.question}")
        
        if not session['show_answer']:
            if st.button("👁️ Mostrar Respuesta", type="primary"):
                session['show_answer'] = True
                session['show_time'] = time.time()
                st.rerun()
        else:
            # Mostrar respuesta
            st.markdown(f"**Respuesta:** {card.answer}")
            
            st.markdown("---")
            st.subheader("¿Cómo fue tu respuesta?")
            
            # Predecir intervalos para todas las opciones
            predictions = predict_intervals_for_all_grades(card, state.params)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.button("❌ Again", use_container_width=True, key="btn_again",
                         on_click=process_review, args=(card, 0, session))
                st.caption(f"🔵 Anki+UIR: **{predictions['anki_uir'][0]}d**")
                st.caption(f"⚪ Anki: {predictions['anki_classic'][0]}d")
            with col2:
                st.button("😓 Hard", use_container_width=True, key="btn_hard",
                         on_click=process_review, args=(card, 1, session))
                st.caption(f"🔵 Anki+UIR: **{predictions['anki_uir'][1]}d**")
                st.caption(f"⚪ Anki: {predictions['anki_classic'][1]}d")
            with col3:
                st.button("✅ Good", use_container_width=True, key="btn_good",
                         on_click=process_review, args=(card, 2, session))
                st.caption(f"🔵 Anki+UIR: **{predictions['anki_uir'][2]}d**")
                st.caption(f"⚪ Anki: {predictions['anki_classic'][2]}d")
            with col4:
                st.button("🌟 Easy", use_container_width=True, key="btn_easy",
                         on_click=process_review, args=(card, 3, session))
                st.caption(f"🔵 Anki+UIR: **{predictions['anki_uir'][3]}d**")
                st.caption(f"⚪ Anki: {predictions['anki_classic'][3]}d")
            
            # Mostrar info de la tarjeta
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("UIC Local", f"{card.UIC_local:.3f}")
            with col_b:
    session['current_card_idx'] += 1
    session['show_answer'] = False
    session['start_time'] = time.time()

def page_analytics():
    """Página de Analytics avanzado con comparación temporal"""
    st.title("📊 Analytics Avanzado")
    
    if not state.cards:
        st.warning("No hay tarjetas para analizar.")
        return
    
    # Selector de período para comparación temporal
    st.sidebar.markdown("### 🕐 Comparación Temporal")
    period = st.sidebar.selectbox("Período", ["Última semana", "Último mes", "Últimos 3 meses"])
    
    # Calcular fechas
    today = datetime.now()
    if period == "Última semana":
        days_back = 7
    elif period == "Último mes":
        days_back = 30
    else:
        days_back = 90
    
    cutoff_date = today - timedelta(days=days_back)
    
    # Tabs para organizar analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Métricas", "🎯 Retención por Tag", "⚠️ Problemáticas", "📅 Predicción"])
    
    with tab1:
        st.subheader("Métricas Generales")
        
        # Métricas actuales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_uir_current = np.mean([c.UIR_effective for c in state.cards])
            st.metric("UIR Promedio", f"{avg_uir_current:.1f}d")
        
        with col2:
            avg_uic_current = np.mean([c.UIC_local for c in state.cards])
            st.metric("UIC Promedio", f"{avg_uic_current:.3f}")
        
        with col3:
            # Tasa de éxito global
            all_grades = []
            for c in state.cards:
                for r in c.history:
                    grade = r.grade if hasattr(r, 'grade') else r.get('grade', 0)
                    all_grades.append(grade)
            
            if all_grades:
                success_rate = sum(1 for g in all_grades if g >= 2) / len(all_grades)
                st.metric("Tasa de Éxito", f"{success_rate*100:.1f}%")
            else:
                st.metric("Tasa de Éxito", "N/A")
        
        with col4:
            # Repasos totales
            total_reviews = sum(len(c.history) for c in state.cards)
            st.metric("Repasos Totales", total_reviews)
        
        # Comparación temporal (deltas)
        st.markdown("---")
        st.subheader(f"Cambios en {period}")
        
        # Calcular métricas del período anterior
        old_reviews = []
        recent_reviews = []
        
        for card in state.cards:
            for review in card.history:
                timestamp = review.timestamp if hasattr(review, 'timestamp') else review.get('timestamp', '')
                if timestamp:
                    review_date = datetime.fromisoformat(timestamp)
                    if review_date >= cutoff_date:
                        recent_reviews.append(review)
                    else:
                        old_reviews.append(review)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            # Delta en repasos
            delta_reviews = len(recent_reviews)
            st.metric("Repasos en Período", delta_reviews)
        
        with col_b:
            # Delta en tasa de éxito
            if recent_reviews:
                recent_success = sum(1 for r in recent_reviews 
                                   if (r.grade if hasattr(r, 'grade') else r.get('grade', 0)) >= 2) / len(recent_reviews)
                st.metric("Éxito Reciente", f"{recent_success*100:.1f}%")
            else:
                st.metric("Éxito Reciente", "N/A")
        
        with col_c:
            # Promedio de repasos por día
            if recent_reviews:
                avg_per_day = len(recent_reviews) / days_back
                st.metric("Repasos/Día", f"{avg_per_day:.1f}")
            else:
                st.metric("Repasos/Día", "0")
        
        # Gráfica de tendencia
        if recent_reviews:
            df_recent = pd.DataFrame([{
                'date': datetime.fromisoformat(r.timestamp if hasattr(r, 'timestamp') else r.get('timestamp', '')).date(),
                'grade': r.grade if hasattr(r, 'grade') else r.get('grade', 0)
            } for r in recent_reviews if hasattr(r, 'timestamp') or 'timestamp' in r])
            
            daily_stats = df_recent.groupby('date').agg({
                'grade': ['count', 'mean']
            }).reset_index()
            daily_stats.columns = ['date', 'count', 'avg_grade']
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['count'],
                                          mode='lines+markers', name='Repasos por Día'))
            fig_trend.update_layout(title=f"Tendencia de Actividad ({period})",
                                   xaxis_title="Fecha", yaxis_title="Repasos")
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        st.subheader("Retención por Tag")
        
        # Agrupar por tags
        tag_stats = {}
        for card in state.cards:
            for tag in card.tags:
                if tag not in tag_stats:
                    tag_stats[tag] = {'cards': 0, 'reviews': 0, 'successes': 0}
                
                tag_stats[tag]['cards'] += 1
    
    # Si es "Again" (grade=0), volver a agregar la tarjeta al final de la cola
    if grade == 0:
        current_card_idx = session["cards_to_review"][session["current_card_idx"]]
        session["cards_to_review"].append(current_card_idx)
                tag_stats[tag]['reviews'] += len(card.history)
                
                for r in card.history:
                    grade = r.grade if hasattr(r, 'grade') else r.get('grade', 0)
                    if grade >= 2:
                        tag_stats[tag]['successes'] += 1
        
        if tag_stats:
            # Calcular tasa de éxito por tag
            tag_data = []
            for tag, stats in tag_stats.items():
                success_rate = (stats['successes'] / stats['reviews'] * 100) if stats['reviews'] > 0 else 0
                tag_data.append({
                    'Tag': tag,
                    'Tarjetas': stats['cards'],
                    'Repasos': stats['reviews'],
                    'Tasa de Éxito (%)': success_rate
                })
            
            df_tags = pd.DataFrame(tag_data).sort_values('Tasa de Éxito (%)', ascending=False)
            st.dataframe(df_tags, use_container_width=True)
            
            # Gráfica de barras
            fig_tags = px.bar(df_tags, x='Tag', y='Tasa de Éxito (%)',
                             title="Tasa de Éxito por Tag",
                             color='Tasa de Éxito (%)',
                             color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_tags, use_container_width=True)
        else:
            st.info("No hay tags para analizar.")
    
    with tab3:
        st.subheader("Tarjetas Problemáticas")
        
        # Identificar tarjetas con baja tasa de éxito
        problematic = []
        for card in state.cards:
            if len(card.history) >= 3:  # Al menos 3 repasos
                recent = card.history[-5:]
                successes = sum(1 for r in recent 
                              if (r.grade if hasattr(r, 'grade') else r.get('grade', 0)) >= 2)
                success_rate = successes / len(recent)
                
                if success_rate < 0.5:  # Menos de 50% de éxito
                    problematic.append({
                        'Pregunta': card.question[:60],
                        'Repasos': len(card.history),
                        'Éxito (%)': success_rate * 100,
                        'UIR': card.UIR_effective,
                        'UIC': card.UIC_local
                    })
        
        if problematic:
            df_prob = pd.DataFrame(problematic).sort_values('Éxito (%)')
            st.dataframe(df_prob, use_container_width=True)
            
            st.markdown("### 💡 Recomendaciones")
            st.info(f"**{len(problematic)} tarjetas** necesitan atención. Considera:\n"
                   "- Revisar si la pregunta/respuesta es clara\n"
                   "- Dividir en tarjetas más simples\n"
                   "- Agregar contexto o ejemplos\n"
                   "- Conectar con otras tarjetas (mejorar UIC)")
        else:
            st.success("✅ No hay tarjetas problemáticas. ¡Buen trabajo!")
    
    with tab4:
        st.subheader("Predicción de Carga de Trabajo")
        
        # Predecir repasos en próximos días
        prediction_days = st.slider("Días a predecir", 7, 90, 30)
        
        workload = {i: 0 for i in range(prediction_days)}
        
        for card in state.cards:
            if card.next_review:
                try:
                    next_dt = datetime.fromisoformat(card.next_review)
                    days_until = (next_dt - today).days
                    
                    if 0 <= days_until < prediction_days:
                        workload[days_until] += 1
                except:
                    pass
        
        df_workload = pd.DataFrame({
            'Día': list(workload.keys()),
            'Repasos Esperados': list(workload.values())
        })
        
        fig_workload = px.bar(df_workload, x='Día', y='Repasos Esperados',
                             title=f"Carga de Trabajo Proyectada ({prediction_days} días)",
                             labels={'Día': 'Días desde hoy'})
        st.plotly_chart(fig_workload, use_container_width=True)
        
        # Estadísticas de la predicción
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("Total Proyectado", sum(workload.values()))
        with col_y:
            st.metric("Promedio/Día", f"{sum(workload.values())/prediction_days:.1f}")
        with col_z:
            max_day = max(workload, key=workload.get)
            st.metric("Día Pico", f"Día {max_day} ({workload[max_day]} repasos)")

def page_semantic_graph():
    """Visualización del grafo semántico"""
    st.title("🕸️ Grafo Semántico")
    
    if len(state.cards) < 2:
        st.warning("Necesitas al menos 2 tarjetas para construir el grafo.")
        return
    
    if st.button("🔄 Reconstruir Grafo", type="primary"):
        with st.spinner("Calculando TF-IDF y similitudes..."):
            tfidf_matrix, vectorizer = compute_tfidf_from_cards(state.cards)
            if tfidf_matrix is not None:
                state.tfidf_matrix = tfidf_matrix
                state.similarity_matrix = compute_similarity_matrix(tfidf_matrix)
    
    # Heatmap
    st.subheader("Mapa de Calor de Similitudes")
    
    fig = px.imshow(state.similarity_matrix,
                    labels=dict(x="Tarjeta", y="Tarjeta", color="Similitud"),
                    x=[f"C{i}" for i in range(len(state.cards))],
                    y=[f"C{i}" for i in range(len(state.cards))],
                    color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de similitudes
    st.subheader("Pares Más Similares")
    
    pairs = []
    n = len(state.cards)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append({
                'Tarjeta 1': state.cards[i].question[:50],
                'Tarjeta 2': state.cards[j].question[:50],
                'Similitud': state.similarity_matrix[i, j]
            })
    
    df_pairs = pd.DataFrame(pairs).sort_values('Similitud', ascending=False).head(10)
    st.dataframe(df_pairs, use_container_width=True)
    
    # Grafo interactivo
    st.subheader("Grafo Interactivo")
    
    threshold = st.slider("Umbral de similitud", 0.0, 1.0, 0.3, 0.05)
    
    # Crear grafo con NetworkX
    G = nx.Graph()
    
    for i, card in enumerate(state.cards):
        G.add_node(i, label=card.question[:30], title=card.question,
                  size=10 + card.UIC_local * 20)
    
    for i in range(n):
        for j in range(i+1, n):
            if state.similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=state.similarity_matrix[i, j])
    
    # Visualizar con pyvis
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.save_graph("data/graph.html")
    
    with open("data/graph.html", 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    st.components.v1.html(html_content, height=600)

def page_algorithm_comparison():
    """Comparación de algoritmos de scheduling"""
    st.title("⚖️ Comparador de Algoritmos")
    
    if not state.cards:
        st.warning("No hay tarjetas para comparar.")
        return
    
    st.subheader("Parámetros del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alpha = st.number_input("Alpha (α)", 0.0, 1.0, state.params['alpha'], 0.05)
    with col2:
        gamma = st.number_input("Gamma (γ)", 0.0, 1.0, state.params['gamma'], 0.05)
    with col3:
        delta = st.number_input("Delta (δ)", 0.0, 1.0, state.params['delta'], 0.01)
    with col4:
        eta = st.number_input("Eta (η)", 0.0, 1.0, state.params['eta'], 0.01)
    
    if st.button("Actualizar Parámetros"):
        state.params.update({'alpha': alpha, 'gamma': gamma, 'delta': delta, 'eta': eta})
        save_state(state)
        st.success("Parámetros actualizados!")
    
    st.markdown("---")
    st.subheader("Comparación de Intervalos")
    
    # Tabla comparativa
    comparison_data = []
    
    for card in state.cards[:20]:  # Limitar a 20 para no saturar
        # Simular próximo intervalo para cada algoritmo
        # (sin modificar la tarjeta real)
        card_copy = Card(**asdict(card))
        
        # Anki clásico
        i_classic = anki_classic_schedule(card_copy, 2)  # Asumiendo "Good"
        
        # Anki+UIR
        card_copy2 = Card(**asdict(card))
        i_uir = anki_uir_adapted_schedule(card_copy2, 2, state.params)
        
        comparison_data.append({
            'Pregunta': card.question[:40],
            'Anki Clásico (días)': i_classic,
            'Anki+UIR (días)': i_uir,
            'Diferencia (%)': ((i_uir - i_classic) / i_classic * 100) if i_classic > 0 else 0,
            'UIC Local': f"{card.UIC_local:.3f}",
            'UIR Efectivo': f"{card.UIR_effective:.1f}"
        })
    
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True)
    
    # Gráfica de distribución de intervalos
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_comp['Anki Clásico (días)'], name='Anki Clásico', opacity=0.7))
    fig.add_trace(go.Histogram(x=df_comp['Anki+UIR (días)'], name='Anki+UIR', opacity=0.7))
    fig.update_layout(title="Distribución de Intervalos Recomendados",
                     xaxis_title="Días", yaxis_title="Frecuencia", barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)

def page_simulation():
    """Simulación de sesiones de repaso"""
    st.title("🔬 Simulación")
    
    st.subheader("Configuración de Simulación")
    
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.number_input("Horizonte (días)", 1, 365, 180)
    with col2:
        algorithm = st.selectbox("Algoritmo", ["Anki Clásico", "Anki+UIR"])
    
    if st.button("▶️ Ejecutar Simulación"):
        if not state.cards:
            st.warning("No hay tarjetas para simular.")
            return
        
        with st.spinner("Simulando..."):
            # Simulación simple: contar repasos por día
            daily_reviews = {i: 0 for i in range(horizon)}
            
            for card in state.cards:
                card_copy = Card(**asdict(card))
                current_day = 0
                
                while current_day < horizon:
                    # Simular repaso con probabilidad
                    grade = np.random.choice([0, 1, 2, 3], p=[0.05, 0.15, 0.5, 0.3])
                    
                    if algorithm == "Anki Clásico":
                        interval = anki_classic_schedule(card_copy, grade)
                    else:
                        interval = anki_uir_adapted_schedule(card_copy, grade, state.params)
                    
                    daily_reviews[current_day] += 1
                    current_day += interval
            
            # Visualizar
            df_sim = pd.DataFrame({
                'Día': list(daily_reviews.keys()),
                'Repasos': list(daily_reviews.values())
            })
            
            fig = px.line(df_sim, x='Día', y='Repasos', 
                         title=f"Repasos por Día - {algorithm}")
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Total de Repasos", sum(daily_reviews.values()))

def page_calibration():
    """Calibración de parámetros"""
    st.title("🎛️ Calibración de Parámetros")
    
    st.subheader("Parámetros Actuales")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Alpha (α)", f"{state.params['alpha']:.3f}")
    with col2:
        st.metric("Gamma (γ)", f"{state.params['gamma']:.3f}")
    with col3:
        st.metric("Delta (δ)", f"{state.params['delta']:.3f}")
    with col4:
        st.metric("Eta (η)", f"{state.params['eta']:.3f}")
    
    st.markdown("---")
    
    # Recopilar historial de repasos
    all_reviews = []
    for card in state.cards:
        all_reviews.extend(card.history)
    
    st.write(f"**Total de repasos registrados:** {len(all_reviews)}")
    
    if len(all_reviews) < 10:
        st.warning("Necesitas al menos 10 repasos para calibrar los parámetros.")
        return
    
    if st.button("🔧 Calibrar Parámetros"):
        with st.spinner("Optimizando parámetros..."):
            # Calibración simple usando grid search
            best_params = state.params.copy()
            # En una implementación real, usaríamos scipy.optimize
            st.info("Calibración completada (placeholder - implementar optimización real)")
            
    st.markdown("---")
    st.subheader("Historial de Repasos")
    
    if all_reviews:
        df_history = pd.DataFrame([{
            'Timestamp': r.timestamp,
            'Grade': r.grade,
            'Response Time': f"{r.response_time:.1f}s",
            'P_recall': f"{r.P_recall:.2f}"
        } for r in all_reviews[-20:]])  # Últimos 20
        
        st.dataframe(df_history, use_container_width=True)

def page_export_import():
    """Export e import de datos"""
    st.title("💾 Export / Import")
    
    tab1, tab2 = st.tabs(["📤 Export", "📥 Import"])
    
    with tab1:
        st.subheader("Exportar Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Exportar Tarjetas (CSV)", use_container_width=True):
                if state.cards:
                    df_export = pd.DataFrame([{
                        'question': c.question,
                        'answer': c.answer,
                        'tags': ','.join(c.tags),
                        'review_count': c.review_count,
                        'UIC_local': c.UIC_local,
                        'UIR_effective': c.UIR_effective
                    } for c in state.cards])
                    
                    csv = df_export.to_csv(index=False)
                    st.download_button("⬇️ Descargar CSV", csv, "cards_export.csv", "text/csv")
                else:
                    st.warning("No hay tarjetas para exportar.")
        
        with col2:
            if st.button("Exportar Estado Completo (JSON)", use_container_width=True):
                state_dict = {
                    'cards': [asdict(card) for card in state.cards],
                    'params': state.params,
                    'last_updated': datetime.now().isoformat()
                }
                
                json_str = json.dumps(state_dict, indent=2, ensure_ascii=False)
                st.download_button("⬇️ Descargar JSON", json_str, "state_export.json", "application/json")
    
    with tab2:
        st.subheader("Importar Estado")
        
        uploaded_json = st.file_uploader("Selecciona archivo JSON de estado", type=['json'])
        
        if uploaded_json:
            if st.button("⚠️ Importar y Reemplazar Estado Actual"):
                try:
                    state_dict = json.load(uploaded_json)
                    
                    # Reconstruir estado
                    cards = [Card(**card_data) for card_data in state_dict.get('cards', [])]
                    for card in cards:
                        card.history = [ReviewHistory(**h) if isinstance(h, dict) else h 
                                       for h in card.history]
                    
                    state.cards = cards
                    state.params = state_dict.get('params', state.params)
                    
                    save_state(state)
                    st.success("✅ Estado importado exitosamente!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importando estado: {e}")

# ============================================================================
# MAIN APP ROUTING
# ============================================================================

current_page = st.session_state.current_page

if current_page == "Dashboard":
    page_dashboard()
elif current_page == "Crear/Importar Tarjetas":
    page_import()
elif current_page == "Sesión de Repaso":
    page_review_session()
elif current_page == "📊 Analytics":
    page_analytics()
elif current_page == "Grafo Semántico":
    page_semantic_graph()
elif current_page == "Comparador de Algoritmos":
    page_algorithm_comparison()
elif current_page == "Simulación":
    page_simulation()
elif current_page == "Calibración":
    page_calibration()
elif current_page == "Export/Import":
    page_export_import()

