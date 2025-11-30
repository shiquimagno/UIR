# Manual Técnico Detallado del Sistema Spaced Repetition

Este documento proporciona una explicación detallada de cada módulo del sistema, sus partes constituyentes y sus funcionalidades.

## Estructura General

El sistema está construido principalmente en Python utilizando **Streamlit** para la interfaz de usuario. La lógica se divide en dos archivos principales:

1.  **`app.py`**: El núcleo de la aplicación. Contiene la lógica de negocio, algoritmos de repaso espaciado (Anki y UIR/UIC), gestión de estado y la interfaz de usuario principal.
2.  **`auth.py`**: Módulo encargado de la autenticación, registro y gestión de sesiones de usuarios.

---

## Módulo 1: `app.py` (Núcleo del Sistema)

Este es el archivo principal que orquesta toda la aplicación. Se puede dividir en varias secciones funcionales:

### 1. Estructuras de Datos (Data Classes)
Definen cómo se modela la información en el sistema.

*   **`ReviewHistory`**: Representa un único evento de repaso de una tarjeta.
    *   *Campos*: `timestamp` (fecha), `grade` (calificación 0-3), `interval` (nuevo intervalo), `ease` (factor de facilidad), `time_taken` (tiempo de respuesta).
    *   *Funcionalidad*: Almacena el historial de rendimiento para calcular métricas futuras.
*   **`Card`**: Representa una tarjeta de estudio.
    *   *Campos*: `question`, `answer`, `tags`, `history` (lista de `ReviewHistory`).
    *   *Metadatos UIR*: `UIC_local` (conectividad semántica), `UIR_base` (retención base), `UIR_effective` (retención real).
    *   *Metadatos Anki*: `easiness_factor`, `interval_days`, `repetition_count`.
*   **`AppState`**: El estado global de la aplicación.
    *   *Campos*: `cards` (lista de todas las tarjetas), `params` (parámetros del modelo UIR como alpha, gamma), `tfidf_matrix` (matriz de características de texto).
*   **`User`**: Representación simple del usuario (aunque la gestión principal está en `auth.py`).

### 2. Persistencia y Gestión de Datos
Funciones encargadas de guardar y cargar la información para que no se pierda al cerrar la aplicación.

*   **`save_state(state)`**: Guarda el objeto `AppState` completo en un archivo JSON. Realiza una copia de seguridad del estado anterior antes de sobrescribir.
*   **`load_state()`**: Lee el archivo JSON y reconstruye los objetos Python (`Card`, `ReviewHistory`, etc.).
*   **`auto_backup()`**: Se ejecuta al inicio para crear copias de seguridad diarias automáticas en `data/auto_backups/`.

### 3. Algoritmos Core: UIR/UIC (Teoría de Interconexión)
Implementan la lógica matemática única de este sistema, basada en la similitud semántica.

*   **`compute_tfidf(cards_data)`**:
    *   *Función*: Convierte el texto de las tarjetas en vectores numéricos usando TF-IDF.
    *   *Detalle*: Utiliza una lista personalizada de **Stop Words** en español (definida en `get_spanish_stop_words`) para filtrar palabras comunes (el, la, que, etc.) y centrarse en el contenido semántico real.
*   **`compute_embeddings(cards_data)`**:
    *   *Función*: (Opcional) Usa modelos de lenguaje (Sentence Transformers) para obtener representaciones semánticas más profundas que TF-IDF.
*   **`compute_similarity_matrix(tfidf_matrix)`**:
    *   *Función*: Calcula qué tan similar es cada tarjeta con todas las demás usando la similitud del coseno.
*   **`compute_UIC_local(W, card_idx)`**:
    *   *Función*: Calcula el **Coeficiente de Interconexión Universal (UIC)** para una tarjeta específica.
    *   *Lógica*: Promedia la similitud de la tarjeta con sus vecinos más cercanos. Un UIC alto significa que la tarjeta está muy conectada con otros conceptos.
*   **`update_on_review(card, grade, ...)`**:
    *   *Función*: Actualiza los valores UIR/UIC de una tarjeta después de un repaso.
    *   *Lógica*: Si recuerdas bien una tarjeta (`grade` alto), su conexión con la red (`UIC`) se fortalece, lo que a su vez mejora su retención base (`UIR_base`).

### 4. Algoritmos de Scheduling (Programación de Repasos)
Determinan cuándo debe repasarse una tarjeta nuevamente.

*   **`anki_classic_schedule(card, grade)`**:
    *   *Función*: Implementación pura del algoritmo SM-2 de Anki.
    *   *Lógica*: Multiplica el intervalo anterior por un factor de facilidad (`EF`).
*   **`anki_uir_adapted_schedule(card, grade, params)`**:
    *   *Función*: El algoritmo híbrido del sistema.
    *   *Lógica*: Calcula el intervalo de Anki y luego lo **modula** usando el factor UIR (`compute_uir_modulation_factor`).
    *   *Efecto*: Si una tarjeta tiene un alto UIC (está muy conectada), el sistema asume que es más difícil de olvidar y extiende el intervalo más allá de lo que haría Anki tradicionalmente.

### 5. Interfaz de Usuario (Pages)
Funciones que renderizan las diferentes pantallas de la aplicación Streamlit.

*   **`page_dashboard()`**: Pantalla principal. Muestra métricas globales (Total tarjetas, UIC Global), gamificación (Nivel, XP, Rachas) y gráficos de actividad.
*   **`page_import()`**: Permite añadir tarjetas.
    *   *Soporte*: Texto manual (`pregunta == respuesta`), CSV, y Markdown de RemNote.
*   **`page_review_session()`**: La interfaz de estudio.
    *   *Funcionalidad*: Muestra la pregunta, permite ver la respuesta y calificar (Again, Hard, Good, Easy).
    *   *Feedback*: Muestra predicciones de cuándo volverás a ver la tarjeta según tu calificación.
*   **`page_analytics()`**: Visualizaciones profundas sobre el aprendizaje, distribución de intervalos y evolución del conocimiento.

---

## Módulo 2: `auth.py` (Autenticación)

Este módulo maneja la seguridad y el acceso de usuarios. A diferencia de sistemas complejos con bases de datos SQL, este utiliza un enfoque ligero basado en archivos JSON.

### Funcionalidades Principales

1.  **Gestión de Usuarios (`users.json`)**:
    *   Almacena un diccionario de usuarios donde la clave es el nombre de usuario.
    *   Guarda el **hash** de la contraseña (SHA-256), nunca la contraseña en texto plano, por seguridad.

2.  **Registro (`register_user`)**:
    *   Verifica que el usuario no exista.
    *   Crea una entrada en `users.json`.
    *   Crea automáticamente un archivo de estado personal para el nuevo usuario (`data/user_{username}_state.json`), inicializándolo vacío.

3.  **Login (`login_user`)**:
    *   Verifica que el usuario exista.
    *   Compara el hash de la contraseña ingresada con el almacenado.
    *   Si es correcto, actualiza la fecha de `last_login`.

4.  **Interfaz de Login (`show_auth_page`)**:
    *   Renderiza las pestañas de "Iniciar Sesión" y "Registrarse" en Streamlit.
    *   Maneja el estado de la sesión (`st.session_state.authenticated`) para bloquear el acceso a `app.py` si no se ha iniciado sesión.

---

## Flujo de Datos Típico

1.  **Inicio**: El usuario abre la app. `app.py` verifica `st.session_state.authenticated`. Si es falso, llama a `auth.show_auth_page()`.
2.  **Login**: El usuario se loguea. `auth.py` valida y marca `authenticated = True`.
3.  **Carga**: `app.py` detecta el usuario y carga su archivo específico `data/user_{username}_state.json` usando `load_state()`.
4.  **Interacción**:
    *   El usuario crea tarjetas en `page_import`.
    *   El sistema calcula similitudes (`compute_tfidf`, `compute_similarity_matrix`) en segundo plano.
5.  **Repaso**:
    *   El usuario entra a `page_review_session`.
    *   Al calificar una tarjeta, se llama a `process_review`.
    *   Se calcula el nuevo intervalo usando `anki_uir_adapted_schedule`.
    *   Se actualizan las métricas UIR/UIC de la tarjeta con `update_on_review`.
6.  **Guardado**: Cada acción crítica llama a `save_state()` para persistir los cambios en el JSON del usuario.
