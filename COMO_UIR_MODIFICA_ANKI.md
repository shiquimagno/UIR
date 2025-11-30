# 🔄 Cómo UIR Modifica el Algoritmo de Anki

## 📌 Resumen Ejecutivo

**Anki Clásico** calcula intervalos usando solo la fórmula SM-2.  
**Anki+UIR** toma ese intervalo y lo **multiplica por un factor** basado en:
- Tu retención individual (UIR)
- Conexiones semánticas (UIC)
- Tu historial reciente
- Tu percepción de dificultad

---

## 🎯 La Fórmula Central

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  I_final = I_anki × Factor_UIR                         │
│                                                         │
│  donde Factor_UIR ∈ [0.5, 2.5]                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Interpretación:**
- `Factor_UIR = 1.0` → Sin cambio (igual que Anki)
- `Factor_UIR = 2.0` → Intervalo 2x más largo
- `Factor_UIR = 0.5` → Intervalo 2x más corto

---

## 📊 Diagrama de Flujo

```
┌─────────────────┐
│  Calificar      │
│  Tarjeta        │
│  (0,1,2,3)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  PASO 1: Calcular Intervalo Anki (SM-2)                │
│                                                         │
│  I_anki = compute_anki_interval_pure(n, EF, I_prev, q) │
│                                                         │
│  Ejemplo: I_anki = 95 días                             │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  PASO 2: Calcular Factor UIR (4 componentes)           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ A. UIR Ratio (Retención Individual)              │  │
│  │    UIR_ratio = UIR_eff / 7.0                     │  │
│  │    Ejemplo: 11.2 / 7.0 = 1.6                     │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ B. UIC Factor (Conexiones Semánticas)           │  │
│  │    UIC_factor = 1 + α × UIC_local                │  │
│  │    Ejemplo: 1 + 0.2 × 0.6 = 1.12                 │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ C. Success Factor (Historial Reciente)          │  │
│  │    success_factor = 0.7 + 0.6 × (éxitos/5)      │  │
│  │    Ejemplo: 0.7 + 0.6 × 1.0 = 1.3                │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ D. Grade Factor (Dificultad Percibida)          │  │
│  │    grade_factor = {0:0.5, 1:0.8, 2:1.0, 3:1.3}  │  │
│  │    Ejemplo: grade=2 → 1.0                        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Factor_UIR = A × B × C × D                            │
│  Factor_UIR = 1.6 × 1.12 × 1.3 × 1.0 = 2.33           │
│  Factor_UIR = clip(2.33, 0.5, 2.5) = 2.33             │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  PASO 3: Aplicar Modulación                            │
│                                                         │
│  I_final = I_anki × Factor_UIR                         │
│  I_final = 95 × 2.33 = 221 días                        │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  PASO 4: Actualizar Tarjeta                            │
│                                                         │
│  card.interval_days = 221                              │
│  card.easiness_factor = EF_new (de Anki)               │
│  card.repetition_count = n_new (de Anki)               │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Análisis Detallado de Cada Componente

### A. UIR Ratio (Retención Individual)

**¿Qué mide?**  
Qué tan bien retienes esta tarjeta comparado con el promedio inicial.

**Fórmula:**
```python
UIR_ratio = card.UIR_effective / UIR_INICIAL
```

**Valores de referencia:**
```
UIR_INICIAL = 7.0 días (constante de referencia)
```

**Tabla de Efectos:**

| UIR_effective | UIR_ratio | Efecto en Intervalo | Interpretación |
|---------------|-----------|---------------------|----------------|
| 3.5 días | 0.5 | ×0.5 (mitad) | Retención muy baja |
| 7.0 días | 1.0 | ×1.0 (neutral) | Retención promedio |
| 10.5 días | 1.5 | ×1.5 (50% más) | Buena retención |
| 14.0 días | 2.0 | ×2.0 (doble) | Excelente retención |

**Ejemplo Práctico:**

```python
# Tarjeta que retienes muy bien
card.UIR_effective = 14.0
UIR_ratio = 14.0 / 7.0 = 2.0
# → Intervalos 2x más largos (base)

# Tarjeta que olvidas rápido
card.UIR_effective = 3.5
UIR_ratio = 3.5 / 7.0 = 0.5
# → Intervalos 2x más cortos (base)
```

**¿Cómo evoluciona UIR_effective?**

Se actualiza después de cada repaso:

```python
# Ecuación de actualización (app.py línea 465-470)
alpha = params['alpha']  # 0.2
card.UIR_effective = card.UIR_base * (1 + alpha * card.UIC_local)

# UIR_base crece con repasos exitosos:
eta = params['eta']  # 0.5
card.UIR_base = card.UIR_base + eta * p_t * card.UIC_local
```

---

### B. UIC Factor (Conexiones Semánticas)

**¿Qué mide?**  
Qué tan conectada está esta tarjeta con otras en tu base de conocimiento.

**Fórmula:**
```python
UIC_factor = 1 + alpha * card.UIC_local
```

**Parámetro:**
```
alpha = 0.2 (peso del refuerzo semántico)
```

**Tabla de Efectos:**

| UIC_local | UIC_factor | Efecto Adicional | Interpretación |
|-----------|------------|------------------|----------------|
| 0.0 | 1.00 | +0% | Tarjeta aislada |
| 0.3 | 1.06 | +6% | Poco conectada |
| 0.6 | 1.12 | +12% | Bien conectada |
| 1.0 | 1.20 | +20% | Muy conectada |

**Ejemplo Práctico:**

```python
# Tarjeta: "¿Qué es la fotosíntesis?"
# Conectada con: clorofila, CO₂, glucosa, luz solar, etc.
card.UIC_local = 0.8
UIC_factor = 1 + 0.2 * 0.8 = 1.16
# → +16% al intervalo

# Tarjeta: "¿Qué es un quark top?"
# Aislada (tema muy específico)
card.UIC_local = 0.1
UIC_factor = 1 + 0.2 * 0.1 = 1.02
# → +2% al intervalo
```

**¿Por qué esto ayuda?**

**Hipótesis:** Conceptos conectados se refuerzan mutuamente.  
Cuando recuerdas "fotosíntesis", también activas:
- Clorofila
- CO₂
- Glucosa
- Luz solar

Esto crea **refuerzo mutuo** → mayor retención → intervalos más largos.

---

### C. Success Factor (Historial Reciente)

**¿Qué mide?**  
Tu desempeño en los últimos 5 repasos de esta tarjeta.

**Fórmula:**
```python
success_rate = (número de Good/Easy en últimos 5) / 5
success_factor = 0.7 + 0.6 * success_rate
```

**Rango:** `[0.7, 1.3]`

**Tabla de Efectos:**

| Últimos 5 Repasos | success_rate | success_factor | Efecto |
|-------------------|--------------|----------------|--------|
| 0 Good/Easy | 0.0 | 0.70 | -30% |
| 1 Good/Easy | 0.2 | 0.82 | -18% |
| 2 Good/Easy | 0.4 | 0.94 | -6% |
| 3 Good/Easy | 0.6 | 1.06 | +6% |
| 4 Good/Easy | 0.8 | 1.18 | +18% |
| 5 Good/Easy | 1.0 | 1.30 | +30% |

**Ejemplo Práctico:**

```python
# Historial: [Good, Good, Good, Good, Good]
success_rate = 5/5 = 1.0
success_factor = 0.7 + 0.6 * 1.0 = 1.3
# → +30% al intervalo

# Historial: [Again, Hard, Again, Good, Hard]
success_rate = 1/5 = 0.2
success_factor = 0.7 + 0.6 * 0.2 = 0.82
# → -18% al intervalo
```

**¿Por qué esto ayuda?**

**Racha positiva** → Confianza en que dominas el concepto → Intervalo más largo  
**Racha negativa** → Señal de dificultad → Intervalo más corto

---

### D. Grade Factor (Dificultad Percibida)

**¿Qué mide?**  
Tu percepción **inmediata** de dificultad en este repaso.

**Fórmula:**
```python
grade_factors = {
    0: 0.5,   # Again
    1: 0.8,   # Hard
    2: 1.0,   # Good
    3: 1.3    # Easy
}
grade_factor = grade_factors[grade]
```

**Tabla de Efectos:**

| Calificación | Nombre | grade_factor | Efecto | Interpretación |
|--------------|--------|--------------|--------|----------------|
| 0 | Again | 0.5 | -50% | No recordé nada |
| 1 | Hard | 0.8 | -20% | Me costó recordar |
| 2 | Good | 1.0 | 0% | Recordé bien |
| 3 | Easy | 1.3 | +30% | Muy fácil |

**Ejemplo Práctico:**

```python
# Calificas "Easy"
grade_factor = 1.3
# → +30% al intervalo (además de otros factores)

# Calificas "Again"
grade_factor = 0.5
# → -50% al intervalo (penalización fuerte)
```

**¿Por qué esto ayuda?**

Tu percepción es un **indicador en tiempo real** de dificultad.  
Anki clásico también usa esto, pero UIR lo **combina con contexto adicional**.

---

## 🧮 Ejemplo Completo Paso a Paso

### Escenario

**Tarjeta:** "¿Qué es la mitocondria?"

**Estado actual:**
```python
card.repetition_count = 5
card.easiness_factor = 2.5
card.interval_days = 38
card.UIR_effective = 11.2
card.UIC_local = 0.6
card.history = [Good, Good, Good, Good, Good]  # Últimos 5
```

**Calificación:** `grade = 2` (Good)

---

### PASO 1: Calcular Intervalo Anki

```python
I_anki, EF_new, n_new = compute_anki_interval_pure(
    n=5, 
    EF=2.5, 
    I_prev=38, 
    grade=2
)

# Para grade=2 (Good) y n≥2:
I_anki = round(I_prev * EF)
I_anki = round(38 * 2.5)
I_anki = 95 días
```

**Resultado Anki Clásico:** `95 días`

---

### PASO 2: Calcular Factor UIR

#### A. UIR Ratio
```python
UIR_ratio = card.UIR_effective / 7.0
UIR_ratio = 11.2 / 7.0
UIR_ratio = 1.6
```

#### B. UIC Factor
```python
UIC_factor = 1 + 0.2 * card.UIC_local
UIC_factor = 1 + 0.2 * 0.6
UIC_factor = 1.12
```

#### C. Success Factor
```python
success_rate = 5/5 = 1.0  # Todos Good
success_factor = 0.7 + 0.6 * 1.0
success_factor = 1.3
```

#### D. Grade Factor
```python
grade_factor = 1.0  # Good
```

#### Combinar
```python
Factor_UIR = UIR_ratio × UIC_factor × success_factor × grade_factor
Factor_UIR = 1.6 × 1.12 × 1.3 × 1.0
Factor_UIR = 2.3296
Factor_UIR = clip(2.3296, 0.5, 2.5)
Factor_UIR = 2.3296  # Dentro del rango
```

---

### PASO 3: Aplicar Modulación

```python
I_final = round(I_anki * Factor_UIR)
I_final = round(95 * 2.3296)
I_final = round(221.312)
I_final = 221 días
```

**Resultado Anki+UIR:** `221 días`

---

### PASO 4: Actualizar Tarjeta

```python
card.interval_days = 221        # Intervalo modulado por UIR
card.easiness_factor = 2.5      # EF de Anki (sin cambio para Good)
card.repetition_count = 6       # n + 1
```

---

### Comparación Final

| Algoritmo | Intervalo | Diferencia |
|-----------|-----------|------------|
| **Anki Clásico** | 95 días | - |
| **Anki+UIR** | 221 días | **+132%** |

**Razón del aumento:**
- ✅ Buena retención individual (UIR=11.2 → ratio 1.6)
- ✅ Tarjeta conectada (UIC=0.6 → factor 1.12)
- ✅ Historial perfecto (5/5 → factor 1.3)
- ✅ Calificación Good (factor 1.0)

**Producto:** `1.6 × 1.12 × 1.3 × 1.0 = 2.33x`

---

## 📉 Ejemplo Opuesto: Tarjeta Difícil

### Escenario

**Tarjeta:** "¿Qué es un tensor de curvatura de Riemann?"

**Estado actual:**
```python
card.repetition_count = 3
card.easiness_factor = 2.2
card.interval_days = 6
card.UIR_effective = 5.2
card.UIC_local = 0.2
card.history = [Again, Hard, Again, Good, Hard]  # Últimos 5
```

**Calificación:** `grade = 1` (Hard)

---

### PASO 1: Calcular Intervalo Anki

```python
# Para grade=1 (Hard):
I_anki = max(1, round(I_prev * 1.2))
I_anki = max(1, round(6 * 1.2))
I_anki = 7 días
```

---

### PASO 2: Calcular Factor UIR

```python
# A. UIR Ratio
UIR_ratio = 5.2 / 7.0 = 0.74

# B. UIC Factor
UIC_factor = 1 + 0.2 * 0.2 = 1.04

# C. Success Factor
success_rate = 1/5 = 0.2  # Solo 1 Good
success_factor = 0.7 + 0.6 * 0.2 = 0.82

# D. Grade Factor
grade_factor = 0.8  # Hard

# Combinar
Factor_UIR = 0.74 × 1.04 × 0.82 × 0.8
Factor_UIR = 0.505
Factor_UIR = clip(0.505, 0.5, 2.5)
Factor_UIR = 0.505
```

---

### PASO 3: Aplicar Modulación

```python
I_final = round(7 * 0.505)
I_final = round(3.535)
I_final = 4 días
```

---

### Comparación Final

| Algoritmo | Intervalo | Diferencia |
|-----------|-----------|------------|
| **Anki Clásico** | 7 días | - |
| **Anki+UIR** | 4 días | **-43%** |

**Razón de la reducción:**
- ❌ Baja retención (UIR=5.2 → ratio 0.74)
- ❌ Tarjeta aislada (UIC=0.2 → factor 1.04)
- ❌ Historial malo (1/5 → factor 0.82)
- ❌ Calificación Hard (factor 0.8)

**Producto:** `0.74 × 1.04 × 0.82 × 0.8 = 0.5x`

---

## 🎯 Límites de Seguridad

### Clipping del Factor UIR

```python
Factor_UIR = np.clip(total_factor, 0.5, 2.5)
```

**¿Por qué?**

| Sin límites | Con límites |
|-------------|-------------|
| Factor podría ser 0.1 → intervalo de 95 días → 9.5 días | Mínimo 0.5 → 47.5 días |
| Factor podría ser 5.0 → intervalo de 95 días → 475 días | Máximo 2.5 → 237.5 días |

**Ventajas:**
- ✅ Evita intervalos extremadamente cortos (frustración)
- ✅ Evita intervalos extremadamente largos (olvido)
- ✅ Mantiene el sistema robusto

---

## 📊 Tabla Resumen de Factores

| Factor | Rango | Efecto Máximo | Cuándo es Alto | Cuándo es Bajo |
|--------|-------|---------------|----------------|----------------|
| **UIR Ratio** | 0.5 - 2.0+ | ±100% | Buena retención | Mala retención |
| **UIC Factor** | 1.0 - 1.2 | +20% | Tarjeta conectada | Tarjeta aislada |
| **Success Factor** | 0.7 - 1.3 | ±30% | Racha positiva | Racha negativa |
| **Grade Factor** | 0.5 - 1.3 | ±50% | Easy | Again |
| **Factor UIR Total** | **0.5 - 2.5** | **±150%** | Todos altos | Todos bajos |

---

## 🔗 Código Fuente

### Función Principal: `anki_uir_adapted_schedule`

**Ubicación:** [`app.py:603-642`](file:///c:/Users/usuario/Desktop/Programación/Spaced%20repetition/app.py#L603-L642)

```python
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
    I_final = max(1, int(I_final))
    
    # 4. Actualizar tarjeta (CRÍTICO: igual que anki_classic_schedule)
    _, EF_new, n_new = compute_anki_interval_pure(
        card.repetition_count,
        card.easiness_factor,
        card.interval_days,
        grade
    )
    
    card.interval_days = I_final
    card.easiness_factor = EF_new
    card.repetition_count = n_new
    
    return I_final
```

### Función de Modulación: `compute_uir_modulation_factor`

**Ubicación:** [`app.py:562-601`](file:///c:/Users/usuario/Desktop/Programación/Spaced%20repetition/app.py#L562-L601)

```python
def compute_uir_modulation_factor(card: Card, grade: int, params: Dict[str, float]) -> float:
    """
    Calcula factor de modulación basado en UIR/UIC
    
    Returns:
        Factor entre 0.5 y 2.5
    """
    UIR_INICIAL = 7.0  # UIR de referencia inicial
    
    # 1. Ratio UIR (progreso de retención)
    UIR_ratio = card.UIR_effective / UIR_INICIAL
    
    # 2. Factor UIC (refuerzo semántico)
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
```

---

## 🎓 Conclusión

### ¿Cómo añade UIR a Anki?

**En una frase:**  
UIR toma el intervalo calculado por Anki y lo **multiplica por un factor inteligente** que considera tu retención individual, conexiones semánticas, historial reciente y percepción de dificultad.

### Ventajas sobre Anki Clásico

| Aspecto | Anki Clásico | Anki+UIR |
|---------|--------------|----------|
| **Personalización** | Genérica (solo EF) | Individual (UIR) |
| **Contexto** | Tarjeta aislada | Red semántica (UIC) |
| **Historial** | Solo cuenta repeticiones | Analiza últimos 5 repasos |
| **Adaptabilidad** | Lenta (solo EF cambia) | Rápida (4 factores) |
| **Robustez** | Puede dar intervalos extremos | Límites [0.5, 2.5] |

### Fórmula Final (Resumen)

```
I_final = I_anki × clip(
    (UIR_eff / 7.0) ×           # Tu retención
    (1 + 0.2 × UIC_local) ×     # Tus conexiones
    (0.7 + 0.6 × success) ×     # Tu historial
    grade_factor,                # Tu percepción
    0.5, 2.5                     # Límites
)
```

---

**Creado:** 2025-11-27  
**Versión:** 1.0  
**Autor:** Sistema UIR/UIC
