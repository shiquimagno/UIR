# Simulador de Spaced Repetition con UIR/UIC

Aplicación interactiva de spaced repetition basada en el paper de Shiqui sobre **Unidades Internacionales de Retención (UIR)** y **Comprensión (UIC)**. Implementa algoritmos avanzados para optimizar el aprendizaje mediante la integración de similitud semántica y retención personalizada.

## 🎯 Características

- **📥 Importación flexible**: Texto, CSV, o formato RemNote
- **🎯 Sesiones de repaso interactivas**: Sistema de calificación (Again/Hard/Good/Easy)
- **🧠 Algoritmos UIR/UIC**: Cálculo dinámico de retención y comprensión
- **🕸️ Grafo semántico**: Visualización de relaciones entre tarjetas con TF-IDF
- **⚖️ Comparación de algoritmos**: Anki clásico vs Anki+UIR
- **🔬 Simulación**: Proyección de repasos a largo plazo
- **🎛️ Calibración**: Optimización de parámetros desde datos reales
- **💾 Persistencia**: Almacenamiento local en JSON con backups automáticos

## 📊 Fundamentos Teóricos

### UIR (Unidad Internacional de Retención)
Mide el tiempo característico de retención de una tarjeta:

```
UIR = -t / ln(P)
```

Donde:
- `t` = tiempo transcurrido desde el último repaso (días)
- `P` = probabilidad de recordar [0,1]

### UIC (Unidad de Comprensión)
Mide la interconexión semántica de una tarjeta con otras:

```
UIC_global = Σ(w_ij) / (n*(n-1))
UIC_local_i = promedio de similitud entre vecinos cercanos
```

Donde `w_ij` es la similitud coseno entre tarjetas i y j.

### Actualización Dinámica

Tras cada repaso:

```python
UIC(t+1) = UIC(t) + γ·p·(1-UIC) - δ·(1-p)·UIC
UIR_base(t+1) = UIR_base(t) + η·p·UIC
UIR_eff = UIR_base · (1 + α·UIC)
```

Parámetros por defecto:
- `α = 0.2` (modulación UIR por UIC)
- `γ = 0.15` (incremento UIC en acierto)
- `δ = 0.02` (decremento UIC en fallo)
- `η = 0.05` (incremento UIR_base)

## 🚀 Instalación

### Requisitos
- Python 3.10 o superior
- pip

### Instalación Local

```bash
# Clonar o descargar el repositorio
cd "Spaced repetition"

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

## 📖 Guía de Uso

### 1. Crear Tarjetas

**Opción A: Desde Texto**
```
¿Qué es Python? == Un lenguaje de programación interpretado
¿Qué es Streamlit? == Framework para crear apps de datos
```

**Opción B: Desde CSV**
```csv
question,answer,tags
¿Qué es una lista?,Estructura de datos mutable,python
```

**Opción C: Importar sample_data.csv**
```bash
# Usa el archivo de ejemplo incluido
```

### 2. Sesión de Repaso

1. Ir a **"Sesión de Repaso"**
2. Click en **"Repasar Pendientes"** o **"Repasar Todas"**
3. Leer la pregunta
4. Click **"Mostrar Respuesta"**
5. Calificar tu respuesta:
   - ❌ **Again**: No recordaste (intervalo reinicia)
   - 😓 **Hard**: Difícil de recordar (intervalo corto)
   - ✅ **Good**: Recordaste bien (intervalo medio)
   - 🌟 **Easy**: Muy fácil (intervalo largo)

### 3. Grafo Semántico

1. Ir a **"Grafo Semántico"**
2. Click **"Reconstruir Grafo"** (calcula TF-IDF y similitudes)
3. Explorar:
   - **Heatmap**: Matriz de similitudes
   - **Tabla**: Pares más similares
   - **Grafo interactivo**: Visualización con pyvis (ajustar umbral)

### 4. Comparador de Algoritmos

- Ver intervalos proyectados para **Anki Clásico** vs **Anki+UIR**
- Ajustar parámetros α, γ, δ, η
- Comparar distribuciones de intervalos

### 5. Simulación

- Configurar horizonte (ej: 180 días)
- Seleccionar algoritmo
- Ver proyección de repasos por día

### 6. Calibración

- Requiere al menos 10 repasos registrados
- Optimiza parámetros desde datos reales
- (Placeholder: implementar scipy.optimize en versión futura)

### 7. Export/Import

**Export:**
- CSV de tarjetas (sin historial)
- JSON completo (incluye historial y parámetros)

**Import:**
- JSON completo para restaurar estado

## 📁 Estructura de Archivos

```
Spaced repetition/
├── app.py                 # Aplicación principal
├── requirements.txt       # Dependencias
├── sample_data.csv        # Datos de ejemplo
├── README.md             # Esta documentación
└── data/                 # Persistencia (creado automáticamente)
    ├── state.json        # Estado actual
    ├── graph.html        # Grafo interactivo
    └── backups/          # Backups automáticos
```

## 🔧 Configuración Avanzada

### Parámetros del Modelo

Editar en **"Comparador de Algoritmos"** o **"Calibración"**:

```python
params = {
    'alpha': 0.2,   # Factor de modulación UIR por UIC
    'gamma': 0.15,  # Tasa de incremento UIC en acierto
    'delta': 0.02,  # Tasa de decremento UIC en fallo
    'eta': 0.05,    # Tasa de incremento UIR_base
}
```

### Personalizar TF-IDF

En `app.py`, función `compute_tfidf()`:

```python
vectorizer = TfidfVectorizer(
    max_features=100,        # Número máximo de términos
    stop_words='spanish',    # Añadir stop words
    ngram_range=(1, 2)       # Unigramas y bigramas
)
```

## 🧪 Testing

### Datos de Ejemplo

```bash
# Importar sample_data.csv desde la UI
# O usar el siguiente código:
```

```python
# En página "Crear/Importar Tarjetas" > CSV
# Subir sample_data.csv
```

### Flujo de Prueba Completo

1. ✅ Importar `sample_data.csv` (10 tarjetas)
2. ✅ Hacer 3 repasos (calificar como "Good")
3. ✅ Reconstruir grafo semántico
4. ✅ Comparar algoritmos
5. ✅ Simular 30 días
6. ✅ Exportar estado


## 📚 Referencias

- Paper de Shiqui sobre UIR/UIC (fundamento teórico)
- Algoritmo Anki/SM-2: [supermemo.com](https://www.supermemo.com/en/archives1990-2015/english/ol/sm2)
- TF-IDF: [scikit-learn.org](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)


## 📄 Licencia

MIT License - Uso libre para proyectos personales y educativos.

## 👤 Autor

Desarrollado como prototipo funcional del sistema UIR/UIC de Shiqui.

---

**¿Preguntas o sugerencias?** Abre un issue en el repositorio.

¡Feliz aprendizaje! 🧠✨
