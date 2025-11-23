# Guía de Despliegue en Streamlit Cloud

## ✅ Código Subido a GitHub

El repositorio está listo en: **https://github.com/shiquimagno/UIR**

Archivos incluidos:
- `app.py` - Aplicación principal
- `requirements.txt` - Dependencias
- `README.md` - Documentación
- `sample_data.csv` - Datos de ejemplo
- `.gitignore` - Archivos excluidos
- `data/.gitkeep` - Estructura de directorios

---

## 🚀 Pasos para Desplegar en Streamlit Cloud

### 1. Acceder a Streamlit Cloud

Ve a: **https://share.streamlit.io**

### 2. Iniciar Sesión

- Click en **"Sign in"**
- Usa tu cuenta de GitHub (shiquimagno)

### 3. Crear Nueva App

1. Click en **"New app"**
2. Selecciona:
   - **Repository:** `shiquimagno/UIR`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Click en **"Deploy!"**

### 4. Configuración Avanzada (Opcional)

Si necesitas configurar algo específico:

- Click en **"Advanced settings"** antes de Deploy
- **Python version:** 3.11 (recomendado)
- **Secrets:** No necesarios para esta app

### 5. Esperar Despliegue

- Streamlit Cloud instalará las dependencias automáticamente
- Proceso toma ~2-3 minutos
- Verás logs en tiempo real

### 6. ¡Listo!

Tu app estará disponible en:
```
https://uir-[random-id].streamlit.app
```

O puedes configurar un nombre personalizado en Settings.

---

## 📋 Checklist de Verificación

Antes de desplegar, asegúrate de que:

- ✅ Repositorio es público
- ✅ `app.py` está en la raíz del repositorio
- ✅ `requirements.txt` tiene todas las dependencias
- ✅ No hay errores de sintaxis en `app.py`

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError"

**Solución:** Verifica que todas las librerías estén en `requirements.txt`

### Error: "FileNotFoundError: data/state.json"

**Normal:** El archivo se crea automáticamente al usar la app

### App muy lenta

**Causa:** Primera carga de TF-IDF con muchas tarjetas
**Solución:** Usar `@st.cache_data` (ya implementado)

### Error de memoria

**Causa:** Grafo muy grande (>100 tarjetas)
**Solución:** Limitar visualización o usar sampling

---

## 🎯 Próximos Pasos Después del Despliegue

1. **Probar la app en producción:**
   - Importar `sample_data.csv`
   - Hacer algunos repasos
   - Verificar que el grafo funciona

2. **Compartir la URL:**
   - Copia la URL de Streamlit Cloud
   - Comparte con usuarios

3. **Monitorear:**
   - Streamlit Cloud muestra analytics básicos
   - Revisa logs si hay errores

4. **Actualizar:**
   - Cualquier push a `main` redespliega automáticamente
   - No necesitas hacer nada manualmente

---

## 📱 Acceso Móvil

La app es responsive y funciona en móviles, pero la experiencia es mejor en desktop para:
- Grafo interactivo
- Tablas grandes
- Visualizaciones complejas

---

## 🔐 Persistencia de Datos

**Importante:** Streamlit Cloud reinicia la app periódicamente, lo que borra `data/state.json`

**Soluciones:**

1. **Corto plazo:** Usar Export/Import JSON regularmente
2. **Largo plazo:** Migrar a base de datos (SQLite en volumen persistente o PostgreSQL)

Para implementar persistencia real:
```python
# Opción 1: Streamlit Secrets + S3/Google Cloud Storage
# Opción 2: Supabase (PostgreSQL gratuito)
# Opción 3: MongoDB Atlas
```

---

## ✨ ¡Listo para Desplegar!

Tu app está lista en GitHub. Solo falta ir a https://share.streamlit.io y seguir los pasos arriba.

**URL del repositorio:** https://github.com/shiquimagno/UIR

¡Buena suerte! 🚀
