# Implementación Técnica: Simulador de Repaso Espaciado con UIR/UIC

## 1. Arquitectura del Sistema

El sistema está implementado como una aplicación web interactiva utilizando **Python** y **Streamlit**. La arquitectura sigue un patrón modular centrado en el estado de la aplicación (`AppState`), que persiste los datos del usuario y la configuración del modelo.

### Componentes Principales
*   **Motor de Scheduling**: Implementa los algoritmos de repaso (Anki Clásico y Anki+UIR).
*   **Modelo Matemático (UIR/UIC)**: Calcula métricas de retención y similitud semántica.
*   **Gestor de Estado**: Maneja la persistencia de datos (tarjetas, historial, parámetros) en formato JSON.
*   **Interfaz de Usuario**: Vistas interactivas para repaso, gestión de tarjetas, simulación y análisis.

## 2. Estructuras de Datos

### 2.1. Tarjeta (`Card`)
La unidad fundamental de información. Cada tarjeta almacena:
*   **Contenido**: Pregunta y respuesta.
*   **Metadatos**: ID único, tags, fecha de creación.
*   **Estado de Repaso**: Intervalo actual, factor de facilidad (EF), conteo de repeticiones.
*   **Métricas UIR/UIC**:
    *   `UIR_base`: Retención intrínseca estimada.
    *   `UIR_effective`: Retención efectiva modulada por el historial.
    *   `UIC_local`: Coeficiente de Interconexión Universal local (similitud con vecinos).
*   **Historial**: Lista de eventos de repaso (`ReviewHistory`).

### 2.2. Historial de Repaso (`ReviewHistory`)
Registra cada interacción del usuario con una tarjeta:
*   `timestamp`: Fecha y hora del repaso.
*   `grade`: Calificación (0=Again, 1=Hard, 2=Good, 3=Easy).
*   `interval`: Intervalo asignado tras el repaso.
*   `ease`: Factor de facilidad en ese momento.
*   `time_taken`: Tiempo de respuesta.

## 3. Algoritmos de Scheduling

### 3.1. Anki Clásico (SM-2 Modificado)
Implementación estándar del algoritmo utilizado por Anki.
*   **Intervalo (I)**: $I_n = I_{n-1} \times EF$
*   **Factor de Facilidad (EF)**: $EF' = EF + (0.1 - (5-q) \times (0.08 + (5-q) \times 0.02))$
    *   Donde $q$ es la calificación (0-3 mapeada a la escala original 0-5).
*   **Lógica de Fallo**: Si $q=0$ (Again), la tarjeta se reinicia o se reduce drásticamente su intervalo.

### 3.2. Algoritmo Híbrido Anki+UIR
Propuesta novedosa que modula el intervalo base de Anki utilizando métricas de la Teoría UIR.

$$I_{final} = I_{Anki} \times M_{UIR}$$

Donde $M_{UIR}$ es el **Factor de Modulación**, calculado como:

$$M_{UIR} = \text{clip}(R_{UIR} \times F_{UIC} \times F_{Success} \times F_{Grade}, 0.5, 2.5)$$

#### Componentes del Factor de Modulación:
1.  **Ratio UIR ($R_{UIR}$)**: Relación entre la retención efectiva de la tarjeta y una referencia inicial.
    $$R_{UIR} = \frac{UIR_{effective}}{UIR_{initial}}$$
    *   *Intuición*: Tarjetas que se retienen mejor de lo esperado pueden espaciarse más.

2.  **Factor UIC ($F_{UIC}$)**: Refuerzo basado en la conexión semántica.
    $$F_{UIC} = 1 + \alpha \times UIC_{local}$$
    *   *Intuición*: Tarjetas semánticamente conectadas con otras (alto UIC) se refuerzan mutuamente, permitiendo intervalos más largos. $\alpha$ es un parámetro calibrable.

3.  **Factor de Éxito ($F_{Success}$)**: Ajuste basado en el historial reciente.
    $$F_{Success} = 0.7 + 0.6 \times \text{TasaExito}$$

4.  **Factor de Dificultad ($F_{Grade}$)**: Ajuste fino según la calificación actual.
    *   Again: 0.5 (penalización fuerte)
    *   Hard: 0.8
    *   Good: 1.0
    *   Easy: 1.3 (bonificación)

## 4. Cálculo de Similitud Semántica (UIC)

Para calcular el **Coeficiente de Interconexión Universal (UIC)**, el sistema construye un grafo semántico de las tarjetas.

1.  **Vectorización TF-IDF**:
    *   Se procesa el texto de todas las tarjetas (pregunta + respuesta).
    *   Se eliminan *stop words* (palabras comunes sin valor semántico).
    *   Se genera una matriz TF-IDF donde cada fila es una tarjeta y cada columna un término.

2.  **Matriz de Similitud**:
    *   Se calcula la similitud coseno entre todos los pares de vectores TF-IDF.
    *   Se obtiene una matriz $W$ de $N \times N$ donde $W_{ij}$ es la similitud entre la tarjeta $i$ y la $j$.
    *   Se anula la diagonal ($W_{ii} = 0$) para ignorar la auto-similitud.

3.  **Cálculo de UIC Local**:
    *   Para cada tarjeta $i$, se calcula el promedio de similitud con sus $k$ vecinos más cercanos (por defecto $k=5$).
    $$UIC_i = \frac{1}{k} \sum_{j \in \text{top-}k(i)} W_{ij}$$

## 5. Simulación y Validación

El sistema incluye un módulo de simulación Monte Carlo para comparar el rendimiento de ambos algoritmos.

*   **Metodología**: Se simula el repaso de un conjunto de tarjetas a lo largo de un horizonte temporal (ej. 180 días).
*   **Modelo de Probabilidad**:
    *   Se asume una distribución de calificaciones (Again, Hard, Good, Easy).
    *   Para **Anki+UIR**, se modela una ligera mejora en las probabilidades de éxito (menos "Again", más "Good") para reflejar el beneficio teórico del refuerzo semántico.
*   **Métricas**:
    *   **Carga de Trabajo**: Número total de repasos necesarios.
    *   **Tarjetas Problemáticas**: Cantidad de tarjetas que caen en ciclos de olvido ("leech").

## 6. Detalles de Implementación en Python

*   **Librerías Clave**:
    *   `scikit-learn`: Para `TfidfVectorizer` y `cosine_similarity`.
    *   `numpy` / `pandas`: Para operaciones matriciales y manejo de datos.
    *   `plotly`: Para visualizaciones interactivas (grafos, líneas de tiempo).
    *   `networkx`: Para visualización de grafos de red.

*   **Optimización**:
    *   Uso de caché (`@st.cache_data`) para operaciones costosas como el cálculo de TF-IDF y matrices de similitud.
    *   Cálculos vectorizados donde es posible para mejorar el rendimiento con grandes volúmenes de tarjetas.
