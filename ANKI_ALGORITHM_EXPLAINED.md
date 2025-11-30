# 🧮 Algoritmo de Anki (SM-2) y su Adaptación al Modelo UIR

## 📚 Índice

1. [¿Qué es el Algoritmo SM-2 de Anki?](#qué-es-el-algoritmo-sm-2-de-anki)
2. [Matemáticas del Algoritmo SM-2](#matemáticas-del-algoritmo-sm-2)
3. [Implementación en tu Código](#implementación-en-tu-código)
4. [Adaptación al Modelo UIR](#adaptación-al-modelo-uir)
5. [Comparación: Anki Clásico vs Anki+UIR](#comparación-anki-clásico-vs-ankiuir)
6. [Ejemplos Prácticos](#ejemplos-prácticos)

---

## ¿Qué es el Algoritmo SM-2 de Anki?

**SM-2** (SuperMemo 2) es el algoritmo de repetición espaciada desarrollado por Piotr Wozniak en 1987. Anki lo adoptó y modificó ligeramente. Su objetivo es **calcular el intervalo óptimo** entre repasos para maximizar la retención a largo plazo.

### Principio Fundamental

> "Cada vez que recuerdas exitosamente algo, el intervalo hasta el próximo repaso debe aumentar exponencialmente."

### Variables Clave

| Variable | Nombre | Descripción | Valor Inicial |
|----------|--------|-------------|---------------|
| `n` | Repetition Count | Número de repasos exitosos consecutivos | 0 |
| `EF` | Easiness Factor | Factor de facilidad (qué tan fácil es la tarjeta) | 2.5 |
| `I` | Interval | Intervalo en días hasta el próximo repaso | 1 |

---

## Matemáticas del Algoritmo SM-2

### Fórmula Original de SM-2

El algoritmo SM-2 calcula el nuevo intervalo `I_new` basándose en:

1. **Calificación del usuario** (`q`): 0-5 en SM-2 original, 0-3 en Anki
2. **Intervalo anterior** (`I_prev`)
3. **Factor de facilidad** (`EF`)

#### Actualización del Factor de Facilidad (EF)

```
EF_new = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
```

**Simplificado en Anki (q = 0-3):**

| Calificación | Nombre | Cambio en EF |
|--------------|--------|--------------|
| 0 | Again | -0.2 |
| 1 | Hard | -0.15 |
| 2 | Good | 0 (sin cambio) |
| 3 | Easy | +0.1 |

**Límite inferior:** `EF ≥ 1.3` (nunca puede ser menor)

#### Cálculo del Nuevo Intervalo

**Caso 1: Again (q = 0)**
```
I_new = 1 día
n_new = 0 (reiniciar contador)
EF_new = max(1.3, EF - 0.2)
```

**Caso 2: Hard (q = 1)**
```
I_new = max(1, round(I_prev * 1.2))
n_new = n + 1
EF_new = max(1.3, EF - 0.15)
```

**Caso 3: Good (q = 2)**
```
Si n = 0:  I_new = 1 día
Si n = 1:  I_new = 6 días
Si n ≥ 2:  I_new = round(I_prev * EF)

n_new = n + 1
EF_new = EF (sin cambio)
```

**Caso 4: Easy (q = 3)**
```
Si n = 0:  I_new = 4 días
Si n ≥ 1:  I_new = round(I_prev * EF * 1.3)

n_new = n + 1
EF_new = EF + 0.1
```

### Ejemplo de Progresión

Supongamos que siempre calificas "Good" (q = 2):

| Repaso | n | EF | I_prev | I_new | Cálculo |
|--------|---|----|----|-------|---------|
| 1 | 0 | 2.5 | - | 1 | Primera vez |
| 2 | 1 | 2.5 | 1 | 6 | Segunda vez |
| 3 | 2 | 2.5 | 6 | 15 | 6 * 2.5 = 15 |
| 4 | 3 | 2.5 | 15 | 38 | 15 * 2.5 = 37.5 ≈ 38 |
| 5 | 4 | 2.5 | 38 | 95 | 38 * 2.5 = 95 |
| 6 | 5 | 2.5 | 95 | 238 | 95 * 2.5 = 237.5 ≈ 238 |

**Observación:** Los intervalos crecen exponencialmente (~2.5x cada vez)

---

## Implementación en tu Código

### Función Pura: `compute_anki_interval_pure`

Esta función calcula el intervalo Anki **sin modificar la tarjeta** (función pura):

```python
def compute_anki_interval_pure(n: int, EF: float, I_prev: int, grade: int) -> Tuple[int, float, int]:
    """
    Calcula intervalo Anki sin modificar la tarjeta (función pura)
    
    Args:
        n: Número de repeticiones exitosas
        EF: Easiness Factor (factor de facilidad)
        I_prev: Intervalo anterior en días
        grade: Calificación (0=Again, 1=Hard, 2=Good, 3=Easy)
    
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
```

### Función con Efectos: `anki_classic_schedule`

Esta función **modifica la tarjeta** in-place:

```python
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
```

---

## Adaptación al Modelo UIR

### El Problema con Anki Clásico

Anki trata **todas las tarjetas por igual**, sin considerar:

- ✗ Conexiones semánticas con otras tarjetas
- ✗ Retención individual del usuario
- ✗ Contexto de aprendizaje

### La Solución: Anki + UIR

Tu implementación **modula el intervalo de Anki** usando métricas UIR/UIC:

```
I_final = I_anki × Factor_UIR
```

Donde:

```
Factor_UIR = clip(
    (UIR_eff / UIR_init) ×           # Progreso de retención
    (1 + α × UIC_local) ×             # Refuerzo semántico
    (0.7 + 0.6 × success_rate) ×     # Historial de éxito
    grade_factor,                     # Dificultad percibida
    0.5, 2.5                          # Límites de seguridad
)
```

### Componentes del Factor UIR

#### 1. **Ratio UIR** (Progreso de Retención)

```python
UIR_ratio = card.UIR_effective / 7.0  # 7.0 = UIR inicial de referencia
```

- Si `UIR_eff = 14.0` → `ratio = 2.0` → intervalos 2x más largos
- Si `UIR_eff = 3.5` → `ratio = 0.5` → intervalos 2x más cortos

**Significado:** Tarjetas que retienes mejor (UIR alto) pueden espaciarse más.

#### 2. **Factor UIC** (Refuerzo Semántico)

```python
UIC_factor = 1 + params['alpha'] * card.UIC_local
```

Con `alpha = 0.2` (default):

| UIC_local | UIC_factor | Efecto |
|-----------|------------|--------|
| 0.0 (aislada) | 1.0 | Sin cambio |
| 0.5 (conectada) | 1.1 | +10% intervalo |
| 1.0 (muy conectada) | 1.2 | +20% intervalo |

**Significado:** Tarjetas conectadas semánticamente se refuerzan mutuamente.

#### 3. **Factor de Éxito** (Historial Reciente)

```python
success_rate = compute_success_rate(card)  # Últimos 5 repasos
success_factor = 0.7 + 0.6 * success_rate  # Rango [0.7, 1.3]
```

| Éxito (últimos 5) | success_rate | success_factor |
|-------------------|--------------|----------------|
| 0/5 | 0.0 | 0.7 (-30%) |
| 3/5 | 0.6 | 1.06 (+6%) |
| 5/5 | 1.0 | 1.3 (+30%) |

**Significado:** Tarjetas con historial positivo reciben intervalos más largos.

#### 4. **Factor de Dificultad** (Calificación Actual)

```python
grade_factors = {
    0: 0.5,   # Again: acortar mucho (-50%)
    1: 0.8,   # Hard: acortar un poco (-20%)
    2: 1.0,   # Good: neutral
    3: 1.3    # Easy: alargar (+30%)
}
```

**Significado:** Tu percepción de dificultad ajusta el intervalo inmediatamente.

### Implementación: `anki_uir_adapted_schedule`

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
    
    # 4. Actualizar tarjeta (igual que Anki clásico)
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

---

## Comparación: Anki Clásico vs Anki+UIR

### Caso 1: Tarjeta Nueva (Primer Repaso "Good")

**Datos:**
- `n = 0`, `EF = 2.5`, `I_prev = 0`
- `grade = 2` (Good)

**Anki Clásico:**
```
I_anki = 1 día (regla fija para n=0)
```

**Anki+UIR:**
```
UIR_eff = 7.0 días (inicial)
UIC_local = 0.0 (sin conexiones aún)
success_rate = 0.5 (neutral)

UIR_ratio = 7.0 / 7.0 = 1.0
UIC_factor = 1 + 0.2*0.0 = 1.0
success_factor = 0.7 + 0.6*0.5 = 1.0
grade_factor = 1.0 (Good)

Factor_UIR = 1.0 × 1.0 × 1.0 × 1.0 = 1.0

I_final = 1 × 1.0 = 1 día
```

**Resultado:** Igual (1 día)

---

### Caso 2: Tarjeta con Historial Positivo (5 Repasos "Good")

**Datos:**
- `n = 5`, `EF = 2.5`, `I_prev = 38`
- `grade = 2` (Good)
- `UIR_eff = 11.2` (incrementado por repasos exitosos)
- `UIC_local = 0.6` (tarjeta conectada)
- `success_rate = 1.0` (5/5 éxitos)

**Anki Clásico:**
```
I_anki = 38 * 2.5 = 95 días
```

**Anki+UIR:**
```
UIR_ratio = 11.2 / 7.0 = 1.6
UIC_factor = 1 + 0.2*0.6 = 1.12
success_factor = 0.7 + 0.6*1.0 = 1.3
grade_factor = 1.0 (Good)

Factor_UIR = 1.6 × 1.12 × 1.3 × 1.0 = 2.33
(clipped a [0.5, 2.5] → 2.33)

I_final = 95 × 2.33 = 221 días
```

**Resultado:** UIR **extiende** el intervalo (221 vs 95 días) ✅

**Razón:** Tarjeta bien aprendida + conectada semánticamente

---

### Caso 3: Tarjeta Difícil (3 Repasos, 2 Fallos)

**Datos:**
- `n = 3`, `EF = 2.2`, `I_prev = 6`
- `grade = 1` (Hard)
- `UIR_eff = 5.2` (bajo por fallos)
- `UIC_local = 0.2` (poco conectada)
- `success_rate = 0.33` (1/3 éxitos)

**Anki Clásico:**
```
I_anki = max(1, round(6 * 1.2)) = 7 días
```

**Anki+UIR:**
```
UIR_ratio = 5.2 / 7.0 = 0.74
UIC_factor = 1 + 0.2*0.2 = 1.04
success_factor = 0.7 + 0.6*0.33 = 0.9
grade_factor = 0.8 (Hard)

Factor_UIR = 0.74 × 1.04 × 0.9 × 0.8 = 0.55

I_final = 7 × 0.55 = 3.85 ≈ 4 días
```

**Resultado:** UIR **acorta** el intervalo (4 vs 7 días) ✅

**Razón:** Tarjeta difícil + aislada semánticamente + historial de fallos

---

## Ejemplos Prácticos

### Ejemplo A: Tarjeta Aislada vs Conectada

**Tarjeta 1:** "¿Qué es un quark top?"
- `UIC_local = 0.1` (aislada, tema muy específico)

**Tarjeta 2:** "¿Qué es la fotosíntesis?"
- `UIC_local = 0.8` (conectada con: clorofila, CO₂, glucosa, etc.)

**Ambas con:**
- `n = 3`, `EF = 2.5`, `I_prev = 15`, `grade = 2` (Good)
- `UIR_eff = 10.0`, `success_rate = 0.8`

**Anki Clásico (ambas):**
```
I_anki = 15 * 2.5 = 38 días
```

**Anki+UIR:**

**Tarjeta 1 (aislada):**
```
UIC_factor = 1 + 0.2*0.1 = 1.02
Factor_UIR = (10/7) × 1.02 × 1.18 × 1.0 = 1.73
I_final = 38 × 1.73 = 66 días
```

**Tarjeta 2 (conectada):**
```
UIC_factor = 1 + 0.2*0.8 = 1.16
Factor_UIR = (10/7) × 1.16 × 1.18 × 1.0 = 1.96
I_final = 38 × 1.96 = 74 días
```

**Diferencia:** La tarjeta conectada obtiene +8 días (12% más)

---

### Ejemplo B: Efecto del Historial

**Tarjeta con 5 repasos:**

**Escenario A:** 5/5 éxitos (success_rate = 1.0)
```
success_factor = 0.7 + 0.6*1.0 = 1.3
```

**Escenario B:** 2/5 éxitos (success_rate = 0.4)
```
success_factor = 0.7 + 0.6*0.4 = 0.94
```

**Diferencia:** Escenario A obtiene intervalos 38% más largos (1.3 / 0.94 ≈ 1.38)

---

## Resumen Ejecutivo

### ¿Estás usando el algoritmo correcto de Anki?

**✅ SÍ**, tu implementación es correcta:

1. **`compute_anki_interval_pure`** implementa SM-2 fielmente
2. **`anki_classic_schedule`** aplica SM-2 puro
3. **`anki_uir_adapted_schedule`** extiende SM-2 con UIR/UIC

### Diferencias Clave

| Aspecto | Anki Clásico | Anki+UIR |
|---------|--------------|----------|
| **Base** | SM-2 puro | SM-2 modulado |
| **Contexto** | Tarjeta aislada | Red semántica |
| **Retención** | EF genérico | UIR individual |
| **Adaptación** | Solo por calificación | Calificación + historial + conexiones |
| **Intervalos** | Fijos por fórmula | Dinámicos por contexto |

### Ventajas del Modelo UIR

1. **Personalización:** Se adapta a tu retención individual
2. **Contexto:** Aprovecha conexiones semánticas
3. **Robustez:** Límites evitan intervalos extremos
4. **Transparencia:** Cada factor es interpretable

### Parámetros Ajustables

```python
params = {
    'alpha': 0.2,    # Peso de UIC (refuerzo semántico)
    'gamma': 0.1,    # Tasa de crecimiento de UIC
    'delta': 0.05,   # Tasa de decaimiento de UIC
    'eta': 0.5       # Tasa de crecimiento de UIR
}
```

---

## Referencias

- **Algoritmo SM-2 Original:** [SuperMemo.com](https://www.supermemo.com/en/archives1990-2015/english/ol/sm2)
- **Documentación Anki:** [Anki Manual - Scheduling](https://docs.ankiweb.net/studying.html)
- **Tu Implementación:**
  - [`app.py:516-642`](file:///c:/Users/usuario/Desktop/Programación/Spaced%20repetition/app.py#L516-L642) - Funciones de scheduling
  - [`UIR_SCHEDULING_MODEL.md`](file:///c:/Users/usuario/Desktop/Programación/Spaced%20repetition/UIR_SCHEDULING_MODEL.md) - Modelo matemático
  - [`UIR_MATHEMATICAL_FOUNDATION.md`](file:///c:/Users/usuario/Desktop/Programación/Spaced%20repetition/UIR_MATHEMATICAL_FOUNDATION.md) - Fundamentos teóricos

---

**Creado:** 2025-11-27  
**Versión:** 1.0
