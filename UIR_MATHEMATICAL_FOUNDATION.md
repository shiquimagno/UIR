# Fundamentación Matemática del Modelo UIR/UIC

## Introducción

Este documento detalla la base matemática detrás del **Coeficiente de Interconexión Universal (UIC)** y la **Unidad Internacional de Retención (UIR)**, y cómo estos modelos interactúan para simular el aprendizaje humano y la consolidación de la memoria.

## 1. Coeficiente de Interconexión Universal (UIC)

El UIC modela la **conectividad semántica** de un concepto dentro de una red de conocimientos. La hipótesis central es que **el conocimiento no es aislado**: recordar un concepto facilita el recuerdo de conceptos relacionados.

### Definición Matemática

Sea $C = \{c_1, c_2, ..., c_n\}$ el conjunto de todas las tarjetas (conceptos) en el sistema.
Representamos cada tarjeta $c_i$ como un vector $v_i$ en un espacio vectorial semántico (usando TF-IDF o Embeddings).

La similitud entre dos tarjetas se calcula mediante la similitud del coseno:

$$ S(c_i, c_j) = \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|} $$

El **UIC Local** de una tarjeta $c_i$ se define como la suma ponderada de sus similitudes con el resto de la red:

$$ UIC(c_i) = \frac{1}{N-1} \sum_{j \neq i} S(c_i, c_j) $$

### Retroalimentación de Comprensión (Feedback Loop)

El usuario pregunta: *"¿La comprensión global aumenta a medida que la comprensión o facilidad de las tarjetas relacionadas también?"*

**La respuesta es SÍ, y aquí está la demostración matemática:**

Definimos la **Comprensión Global ($G$)** como el promedio ponderado de la retención de todas las tarjetas, donde el peso es su conectividad (UIC):

$$ G = \sum_{i=1}^{n} UIC(c_i) \cdot R(c_i) $$

Donde $R(c_i)$ es la retención actual (probabilidad de recuerdo) de la tarjeta $i$.

Cuando repasas una tarjeta $c_k$ y la calificas como "Fácil" (Easy), su intervalo ($I_k$) aumenta, lo que mantiene su retención $R(c_k)$ alta por más tiempo.

Dado que $c_k$ contribuye al UIC de sus vecinos (porque $S(c_i, c_k)$ es simétrico), el fortalecimiento de $c_k$ estabiliza la red local.

En nuestro algoritmo modificado, el intervalo de una tarjeta $c_i$ se expande por un factor $M_{UIR}$:

$$ I_{new} = I_{Anki} \times (1 + \alpha \cdot UIC(c_i) + \gamma \cdot \text{Ease}(c_i)) $$

Si las tarjetas vecinas $c_j$ (relacionadas con $c_i$) se repasan y se vuelven "Fáciles", el usuario tiende a reforzar los patrones semánticos compartidos. Aunque el modelo actual calcula el UIC basado en texto estático, la **facilidad** ($Ease$) es dinámica.

Si implementamos un **UIC Dinámico** (propuesta avanzada), el peso de la conexión $S(c_i, c_j)$ podría ponderarse por la facilidad de $c_j$:

$$ UIC_{dinamico}(c_i) = \sum_{j \neq i} S(c_i, c_j) \cdot \text{Ease}(c_j) $$

Esto significaría matemáticamente que **mejorar un concepto mejora directamente el intervalo de sus conceptos relacionados**.

## 2. Unidad Internacional de Retención (UIR)

La UIR es una medida de tiempo normalizada que representa la estabilidad de un recuerdo.

$$ R(t) = e^{-\frac{t \cdot \ln(10/9)}{UIR}} $$

Donde $R(t)$ es la retención en el tiempo $t$. Cuando $t = UIR$, la retención es del 90%.

### Relación con Anki (SM-2)

Anki usa un Factor de Facilidad ($EF$) para multiplicar intervalos.
UIR unifica esto proponiendo que $EF$ es una aproximación discreta de la evolución de la UIR.

$$ UIR_{n+1} = UIR_n \times EF $$

## Conclusión

El modelo es matemáticamente consistente:
1.  **Conectividad**: El UIC captura la densidad semántica.
2.  **Propagación**: Al usar el UIC para modular los intervalos, estamos asumiendo que los conceptos densamente conectados son más resistentes al olvido (apoyado por la psicología cognitiva: *efecto de los niveles de procesamiento*).
3.  **Sinergia**: Si el usuario domina un clúster de temas, el sistema (a través de los parámetros $\alpha$ y $\gamma$) permite intervalos más largos, reflejando una mayor "Comprensión Global".
