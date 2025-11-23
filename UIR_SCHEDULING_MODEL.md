# 🧮 Modelo Matemático Mejorado: Anki+UIR

## ❌ Problema Actual

El algoritmo `anki_uir_adapted_schedule` tiene un **bug crítico**:

```python
def anki_uir_adapted_schedule(card: Card, grade: int, params: Dict[str, float]) -> int:
    # ❌ PROBLEMA: Modifica la tarjeta original
    I_classic = anki_classic_schedule(card, grade)  
    
    # Escalar por UIR
    F = card.UIR_effective / card.UIR_base
    I_adapted = round(I_classic * F)
    return I_adapted
```

**Consecuencia:** Ambos algoritmos dan el **mismo resultado** porque `anki_classic_schedule` modifica `card.interval_days` y `card.easiness_factor`.

---

## ✅ Solución: Modelo UIR Nativo

### Fundamento Teórico

**Curva de olvido exponencial:**
```
P(t) = exp(-t / UIR_eff)
```

Donde:
- `P(t)` = probabilidad de recordar después de `t` días
- `UIR_eff` = UIR efectivo (modulado por UIC)

**Objetivo:** Encontrar `t` tal que `P(t) = P_target`

```
P_target = exp(-t / UIR_eff)
ln(P_target) = -t / UIR_eff
t = -UIR_eff * ln(P_target)
```

### Modelo Propuesto

#### 1. **Probabilidad Objetivo por Calificación**

```python
grade_to_target_P = {
    0: 0.90,  # Again: quiero 90% de probabilidad al próximo repaso
    1: 0.85,  # Hard: 85% de probabilidad
    2: 0.80,  # Good: 80% de probabilidad (balance óptimo)
    3: 0.70,  # Easy: 70% de probabilidad (puedo arriesgar más)
}
```

**Rationale:**
- **Again:** Necesito repasar pronto con alta certeza
- **Hard:** Todavía difícil, ser conservador
- **Good:** Balance entre retención y eficiencia
- **Easy:** Puedo espaciar más, acepto más riesgo de olvido

#### 2. **Cálculo del Intervalo Base (UIR puro)**

```python
def uir_native_schedule(card: Card, grade: int, params: Dict[str, float]) -> int:
    """
    Algoritmo nativo basado en UIR (no deriva de Anki)
    """
    # Probabilidades objetivo
    grade_to_target_P = {0: 0.90, 1: 0.85, 2: 0.80, 3: 0.70}
    P_target = grade_to_target_P[grade]
    
    # Intervalo base desde UIR efectivo
    UIR_eff = card.UIR_effective
    I_base = -UIR_eff * np.log(P_target)
    
    # Ajuste por historial de éxito
    success_rate = compute_success_rate(card)
    I_adjusted = I_base * (0.5 + success_rate)  # Factor [0.5, 1.5]
    
    # Ajuste por UIC (tarjetas conectadas se refuerzan)
    UIC_boost = 1 + params['alpha'] * card.UIC_local
    I_final = I_adjusted * UIC_boost
    
    return max(1, round(I_final))
```

#### 3. **Ajuste por Historial de Éxito**

```python
def compute_success_rate(card: Card) -> float:
    """
    Tasa de éxito reciente (últimos 5 repasos)
    """
    if not card.history:
        return 0.5  # Neutral para tarjetas nuevas
    
    recent = card.history[-5:]  # Últimos 5
    successes = sum(1 for r in recent if r.grade >= 2)  # Good o Easy
    return successes / len(recent)
```

**Rationale:**
- Tarjetas con historial de éxito → intervalos más largos
- Tarjetas con fallos recientes → intervalos más cortos
- Factor multiplicador: [0.5, 1.5]

#### 4. **Boost por UIC (Refuerzo Semántico)**

```python
UIC_boost = 1 + alpha * UIC_local
```

**Ejemplo:**
- `UIC_local = 0.0` (tarjeta aislada) → `boost = 1.0` (sin cambio)
- `UIC_local = 0.5` (medianamente conectada) → `boost = 1.1` (+10%)
- `UIC_local = 1.0` (muy conectada) → `boost = 1.2` (+20%)

**Rationale:** Tarjetas en clusters semánticos se refuerzan mutuamente (efecto de red).

---

## 📊 Comparación: Anki Clásico vs UIR Nativo

### Caso 1: Tarjeta Nueva (primer repaso "Good")

**Anki Clásico:**
```
n = 0 → I = 1 día
```

**UIR Nativo:**
```
UIR_eff = 7.0 días (inicial)
P_target = 0.80 (Good)
I_base = -7.0 * ln(0.80) = 1.56 días
success_rate = 0.5 (neutral)
I_adjusted = 1.56 * 1.0 = 1.56 días
UIC_boost = 1.0 (sin conexiones aún)
I_final = 1.56 → 2 días
```

**Diferencia:** UIR da intervalo ligeramente más largo (2 vs 1 día)

---

### Caso 2: Tarjeta con Historial Positivo (5 repasos "Good")

**Anki Clásico:**
```
n = 5, EF = 2.5
I_prev = 38 días (acumulado)
I_new = 38 * 2.5 = 95 días
```

**UIR Nativo:**
```
UIR_eff = 12.0 días (incrementado por repasos exitosos)
UIC_local = 0.6 (tarjeta conectada)
P_target = 0.80
I_base = -12.0 * ln(0.80) = 2.68 días
success_rate = 1.0 (5/5 éxitos)
I_adjusted = 2.68 * 1.5 = 4.02 días
UIC_boost = 1 + 0.2*0.6 = 1.12
I_final = 4.02 * 1.12 = 4.5 → 5 días
```

**Problema detectado:** UIR da intervalo MUY corto (5 vs 95 días)

**Razón:** UIR_eff no crece lo suficiente con los repasos.

---

## 🔧 Solución: Modelo Híbrido Mejorado

### Combinar Anki + UIR de Forma Inteligente

```python
def anki_uir_hybrid_schedule(card: Card, grade: int, params: Dict[str, float]) -> int:
    """
    Modelo híbrido que combina lo mejor de Anki y UIR
    """
    # 1. Calcular intervalo Anki (sin modificar card)
    I_anki = compute_anki_interval(card, grade)  # Función pura
    
    # 2. Calcular factor de modulación UIR
    UIR_factor = compute_uir_modulation_factor(card, grade, params)
    
    # 3. Combinar
    I_final = round(I_anki * UIR_factor)
    
    return max(1, I_final)

def compute_uir_modulation_factor(card: Card, grade: int, params: Dict[str, float]) -> float:
    """
    Factor de modulación basado en UIR/UIC
    Rango típico: [0.5, 2.0]
    """
    # Base: ratio UIR_eff / UIR_inicial
    UIR_ratio = card.UIR_effective / 7.0  # 7.0 = UIR inicial
    
    # Ajuste por UIC (tarjetas conectadas → intervalos más largos)
    UIC_factor = 1 + params['alpha'] * card.UIC_local
    
    # Ajuste por tasa de éxito reciente
    success_rate = compute_success_rate(card)
    success_factor = 0.7 + 0.6 * success_rate  # Rango [0.7, 1.3]
    
    # Ajuste por dificultad percibida (grade)
    grade_factors = {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.3}
    grade_factor = grade_factors[grade]
    
    # Combinar todos los factores
    total_factor = UIR_ratio * UIC_factor * success_factor * grade_factor
    
    # Limitar rango para evitar extremos
    return np.clip(total_factor, 0.5, 2.5)
```

---

## 📈 Ejemplo Completo: Modelo Híbrido

### Tarjeta con Historial (5 repasos "Good")

**Datos:**
```
UIR_base = 10.0 días
UIC_local = 0.6
UIR_effective = 10.0 * (1 + 0.2*0.6) = 11.2 días
success_rate = 1.0 (5/5)
grade = 2 (Good)
```

**Anki Clásico:**
```
I_anki = 95 días
```

**Factor UIR:**
```
UIR_ratio = 11.2 / 7.0 = 1.6
UIC_factor = 1 + 0.2*0.6 = 1.12
success_factor = 0.7 + 0.6*1.0 = 1.3
grade_factor = 1.0 (Good)

total_factor = 1.6 * 1.12 * 1.3 * 1.0 = 2.33
clipped = 2.33 (dentro de [0.5, 2.5])
```

**Intervalo Final:**
```
I_final = 95 * 2.33 = 221 días
```

**Resultado:** UIR **extiende** el intervalo Anki (221 vs 95 días) para tarjetas bien aprendidas y conectadas.

---

### Tarjeta Difícil (3 repasos, 2 fallos)

**Datos:**
```
UIR_base = 5.0 días (bajo por fallos)
UIC_local = 0.2 (poco conectada)
UIR_effective = 5.0 * (1 + 0.2*0.2) = 5.2 días
success_rate = 0.33 (1/3)
grade = 1 (Hard)
```

**Anki Clásico:**
```
I_anki = 8 días
```

**Factor UIR:**
```
UIR_ratio = 5.2 / 7.0 = 0.74
UIC_factor = 1 + 0.2*0.2 = 1.04
success_factor = 0.7 + 0.6*0.33 = 0.9
grade_factor = 0.8 (Hard)

total_factor = 0.74 * 1.04 * 0.9 * 0.8 = 0.55
```

**Intervalo Final:**
```
I_final = 8 * 0.55 = 4.4 → 4 días
```

**Resultado:** UIR **acorta** el intervalo Anki (4 vs 8 días) para tarjetas difíciles y aisladas.

---

## 🎯 Resumen del Modelo

### Fórmula Final

```python
I_final = I_anki * UIR_factor

donde:
UIR_factor = clip(
    (UIR_eff / UIR_init) *           # Progreso de retención
    (1 + α * UIC_local) *             # Refuerzo semántico
    (0.7 + 0.6 * success_rate) *     # Historial de éxito
    grade_factor,                     # Dificultad percibida
    0.5, 2.5                          # Límites de seguridad
)
```

### Ventajas

1. **Diferenciación clara:** Anki+UIR ≠ Anki clásico
2. **Adaptativo:** Se ajusta a retención individual (UIR)
3. **Contextual:** Considera conexiones semánticas (UIC)
4. **Robusto:** Límites evitan intervalos extremos
5. **Interpretable:** Cada factor tiene significado claro

### Parámetros Ajustables

- `α` (alpha): Peso de UIC (default 0.2)
- `UIR_init`: UIR inicial (default 7.0 días)
- `grade_factors`: Multiplicadores por dificultad
- `clip_range`: Rango de modulación permitido

---

## 📊 Predicción de Intervalos en UI

### Mostrar Durante Repaso

```
┌─────────────────────────────────────┐
│ ¿Qué es la teoría de cuerdas?      │
│                                     │
│ [Mostrar Respuesta]                │
└─────────────────────────────────────┘

Después de mostrar respuesta:

┌─────────────────────────────────────┐
│ Respuesta: Teoría física que...    │
│                                     │
│ ❌ Again  😓 Hard  ✅ Good  🌟 Easy│
│   1 día    4 días   12 días  30 días│
│                                     │
│ Anki Clásico:                       │
│   1 día    5 días   10 días  25 días│
└─────────────────────────────────────┘
```

**Implementación:**
```python
# Calcular intervalos para cada opción
intervals_uir = {
    0: anki_uir_hybrid_schedule(card_copy, 0, params),
    1: anki_uir_hybrid_schedule(card_copy, 1, params),
    2: anki_uir_hybrid_schedule(card_copy, 2, params),
    3: anki_uir_hybrid_schedule(card_copy, 3, params),
}

# Mostrar debajo de cada botón
st.button(f"❌ Again\n{intervals_uir[0]} días")
```

---

**Estado:** Modelo diseñado, listo para implementar
