# 🔍 Análisis: UIR_INICIAL vs Intervalos de Anki

## 🎯 La Pregunta

> "El intervalo inicial es 7, en comparación a Anki que creo que es 4, ¿no sé si eso sea un error por defecto?"

**Respuesta corta:** No es un error, pero hay una **confusión conceptual** importante que debemos aclarar.

---

## 📊 Comparación de Valores Iniciales

### Anki Clásico - Intervalos Iniciales

Según [`app.py:516-538`](file:///c:/Users/usuario/Desktop/Programación/Spaced%20repetition/app.py#L516-L538):

| Calificación | Primera Vez (n=0) | Segunda Vez (n=1) |
|--------------|-------------------|-------------------|
| **Again (0)** | 1 día | 1 día (reinicia) |
| **Hard (1)** | - | I_prev × 1.2 |
| **Good (2)** | **1 día** | **6 días** |
| **Easy (3)** | **4 días** | I_prev × EF × 1.3 |

**Observación:** 
- Good (la más común) → 1 día
- Easy (optimista) → 4 días
- **Promedio ponderado ≈ 1-2 días** (no 4)

### UIR - Valor Inicial

Según [`app.py:74-75`](file:///c:/Users/usuario/Desktop/Programación/Spaced%20repetition/app.py#L74-L75):

```python
UIR_base: float = 7.0  # días (valor inicial razonable)
UIR_effective: float = 7.0
```

**Observación:** 
- UIR_base = 7.0 días
- UIR_effective = 7.0 días (inicialmente)

---

## ⚠️ La Confusión Conceptual

### Problema: Estamos mezclando dos conceptos diferentes

#### 1. **Intervalo de Anki** (I_anki)
- **Qué es:** Días hasta el próximo repaso
- **Valores iniciales:** 1 día (Good), 4 días (Easy)
- **Unidad:** Días calendario

#### 2. **UIR_INICIAL** (línea 575)
- **Qué es:** Valor de **referencia** para calcular el ratio
- **Valor:** 7.0 días
- **Unidad:** Días de retención (no intervalo)
- **Uso:** Normalización del factor UIR

### Fórmula Actual (línea 575-578)

```python
UIR_INICIAL = 7.0  # UIR de referencia inicial

# 1. Ratio UIR (progreso de retención)
UIR_ratio = card.UIR_effective / UIR_INICIAL
```

**Interpretación:**
- Si `UIR_eff = 7.0` → ratio = 1.0 → **sin cambio**
- Si `UIR_eff = 14.0` → ratio = 2.0 → **intervalos 2x más largos**
- Si `UIR_eff = 3.5` → ratio = 0.5 → **intervalos 2x más cortos**

---

## 🤔 ¿Es 7.0 el Valor Correcto?

### Opción A: Mantener UIR_INICIAL = 7.0

**Ventajas:**
- ✅ Representa ~1 semana de retención (valor psicológicamente razonable)
- ✅ Neutral para tarjetas nuevas (ratio = 1.0)
- ✅ Consistente con `UIR_base` inicial

**Desventajas:**
- ⚠️ No está directamente relacionado con intervalos de Anki
- ⚠️ Puede confundir porque 7 ≠ 1 (intervalo Good) ni 4 (intervalo Easy)

### Opción B: Cambiar a UIR_INICIAL = 1.0

**Ventajas:**
- ✅ Alineado con intervalo inicial de Anki para "Good"
- ✅ Más intuitivo: ratio = UIR_eff / 1.0 = UIR_eff directamente

**Desventajas:**
- ❌ Tarjetas nuevas tendrían ratio = 7.0 → intervalos 7x más largos (demasiado)
- ❌ Requiere reajustar todos los valores de UIR_base

### Opción C: Cambiar a UIR_INICIAL = 4.0

**Ventajas:**
- ✅ Alineado con intervalo "Easy" de Anki
- ✅ Valor intermedio razonable

**Desventajas:**
- ⚠️ Tarjetas nuevas tendrían ratio = 7.0/4.0 = 1.75 → intervalos 75% más largos
- ⚠️ Menos neutral que 7.0

---

## 💡 Recomendación

### Mantener UIR_INICIAL = 7.0, pero **aclarar su significado**

**Razón:** UIR_INICIAL **NO es un intervalo**, es un **valor de referencia de retención**.

### Cambio Sugerido en el Código

```python
# ANTES (línea 575)
UIR_INICIAL = 7.0  # UIR de referencia inicial

# DESPUÉS (más claro)
UIR_REFERENCIA = 7.0  # Valor de referencia para normalización (≈1 semana de retención)
# Este NO es un intervalo de Anki, es el UIR_base inicial de tarjetas nuevas
# Usado para calcular el ratio: UIR_ratio = UIR_eff / UIR_REFERENCIA
```

### Actualización de la Fórmula

```python
def compute_uir_modulation_factor(card: Card, grade: int, params: Dict[str, float]) -> float:
    """
    Calcula factor de modulación basado en UIR/UIC
    
    Returns:
        Factor entre 0.5 y 2.5
    """
    # Valor de referencia: UIR_base inicial de tarjetas nuevas
    # NO confundir con intervalos de Anki (que son 1-4 días)
    # Este valor representa ~1 semana de retención base
    UIR_REFERENCIA = 7.0
    
    # 1. Ratio UIR (progreso de retención)
    # ratio = 1.0 → tarjeta con retención promedio
    # ratio > 1.0 → mejor retención que promedio
    # ratio < 1.0 → peor retención que promedio
    UIR_ratio = card.UIR_effective / UIR_REFERENCIA
    
    # ... resto del código
```

---

## 📈 Ejemplo Numérico: ¿Por qué 7.0 funciona?

### Tarjeta Nueva (primer repaso)

**Estado inicial:**
```python
card.UIR_base = 7.0
card.UIR_effective = 7.0
card.UIC_local = 0.0
```

**Calificación: Good (2)**

**Anki Clásico:**
```python
I_anki = 1 día  # Regla fija para n=0, grade=2
```

**Factor UIR:**
```python
UIR_ratio = 7.0 / 7.0 = 1.0
UIC_factor = 1 + 0.2 * 0.0 = 1.0
success_factor = 0.7 + 0.6 * 0.5 = 1.0  # Neutral
grade_factor = 1.0  # Good

Factor_UIR = 1.0 × 1.0 × 1.0 × 1.0 = 1.0
```

**Resultado:**
```python
I_final = 1 × 1.0 = 1 día
```

**Conclusión:** Para tarjetas nuevas, UIR_INICIAL = 7.0 es **neutral** (no modifica Anki) ✅

---

### Tarjeta con Progreso (5 repasos)

**Estado:**
```python
card.UIR_base = 11.0  # Creció por repasos exitosos
card.UIR_effective = 11.2  # 11.0 × (1 + 0.2×0.1)
card.UIC_local = 0.1
```

**Anki Clásico:**
```python
I_anki = 95 días
```

**Factor UIR:**
```python
UIR_ratio = 11.2 / 7.0 = 1.6
UIC_factor = 1 + 0.2 * 0.1 = 1.02
success_factor = 1.3  # Historial perfecto
grade_factor = 1.0

Factor_UIR = 1.6 × 1.02 × 1.3 × 1.0 = 2.12
```

**Resultado:**
```python
I_final = 95 × 2.12 = 201 días
```

**Conclusión:** UIR_INICIAL = 7.0 permite que tarjetas bien aprendidas **extiendan** intervalos ✅

---

## 🔄 Alternativa: Usar UIR_base Inicial Dinámico

### Problema Actual

Todas las tarjetas empiezan con `UIR_base = 7.0`, independientemente de su dificultad inicial.

### Propuesta Mejorada

Ajustar `UIR_base` inicial según la **primera calificación**:

```python
def initialize_card_uir(card: Card, first_grade: int):
    """
    Inicializa UIR_base según la primera impresión del usuario
    """
    initial_uir_map = {
        0: 3.0,   # Again: muy difícil
        1: 5.0,   # Hard: difícil
        2: 7.0,   # Good: promedio (actual)
        3: 10.0   # Easy: fácil
    }
    
    card.UIR_base = initial_uir_map.get(first_grade, 7.0)
    card.UIR_effective = card.UIR_base
```

**Ventaja:** Tarjetas fáciles desde el inicio obtienen UIR más alto → intervalos más largos desde el principio.

---

## 📊 Tabla Comparativa de Opciones

| Opción | UIR_INICIAL | Tarjeta Nueva (ratio) | Tarjeta Progresada (UIR=14) | Pros | Contras |
|--------|-------------|----------------------|----------------------------|------|---------|
| **Actual** | 7.0 | 1.0 (neutral) | 2.0 (2x) | Neutral, consistente | Confuso vs Anki |
| **Opción 1** | 1.0 | 7.0 (7x más largo) | 14.0 (14x) | Alineado con Anki Good | Demasiado agresivo |
| **Opción 4** | 4.0 | 1.75 (75% más) | 3.5 (3.5x) | Alineado con Anki Easy | Menos neutral |
| **Opción UIR_base** | 7.0 | 1.0 (neutral) | 2.0 (2x) | Dinámico por dificultad | Más complejo |

---

## ✅ Decisión Recomendada

### Mantener UIR_INICIAL = 7.0

**Razones:**

1. **Semántica correcta:** UIR_INICIAL es un **valor de retención**, no un intervalo
2. **Neutralidad:** Tarjetas nuevas tienen ratio = 1.0 (no modifican Anki)
3. **Consistencia:** Alineado con `UIR_base` inicial (línea 74)
4. **Escalabilidad:** Permite que UIR crezca/decrezca naturalmente

### Mejoras Sugeridas

1. **Renombrar para claridad:**
   ```python
   UIR_REFERENCIA = 7.0  # Valor de referencia (NO es intervalo de Anki)
   ```

2. **Documentar mejor:**
   ```python
   # UIR_REFERENCIA representa el UIR_base inicial de tarjetas nuevas
   # Es el punto de referencia para calcular el ratio de modulación
   # ratio = 1.0 → retención promedio (sin modificar Anki)
   # ratio > 1.0 → mejor retención (intervalos más largos)
   # ratio < 1.0 → peor retención (intervalos más cortos)
   ```

3. **Opcional - UIR inicial dinámico:**
   Ajustar `UIR_base` según primera calificación (ver propuesta arriba)

---

## 🎓 Conclusión

**No es un error.** UIR_INICIAL = 7.0 es correcto porque:

- ✅ Representa un valor de **retención** (no intervalo)
- ✅ Es neutral para tarjetas nuevas (ratio = 1.0)
- ✅ Permite escalamiento natural con el progreso
- ✅ Consistente con el modelo UIR/UIC

**La confusión viene de comparar:**
- Intervalos de Anki (1-4 días) ← Días calendario
- UIR_INICIAL (7.0 días) ← Días de retención (métrica diferente)

**Son conceptos diferentes que no deben compararse directamente.**

---

## 📝 Cambios Propuestos al Código

### Cambio Mínimo (Solo Documentación)

```python
def compute_uir_modulation_factor(card: Card, grade: int, params: Dict[str, float]) -> float:
    """
    Calcula factor de modulación basado en UIR/UIC
    
    Returns:
        Factor entre 0.5 y 2.5
    """
    # Valor de referencia para normalización del ratio UIR
    # Representa el UIR_base inicial de tarjetas nuevas (~1 semana de retención)
    # NOTA: Este NO es un intervalo de Anki (que son 1-4 días)
    #       Es una métrica de retención diferente
    UIR_REFERENCIA = 7.0
    
    # 1. Ratio UIR (progreso de retención)
    # Compara la retención actual vs la retención inicial de referencia
    UIR_ratio = card.UIR_effective / UIR_REFERENCIA
    
    # ... resto del código
```

---

**Creado:** 2025-11-27  
**Versión:** 1.0
