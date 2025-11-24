# Cambios pendientes para app.py

## 1. Eliminar modo claro (solo oscuro)
- Líneas 715-765: Eliminar todo el código del toggle y modo claro
- Dejar solo el título del sidebar

## 2. Arreglar "Again" para que vuelva a la cola
- Función process_review (línea ~1359)
- Agregar lógica: if grade == 0, append card index to session['cards_to_review']

## Implementación segura:
- Hacer cambios uno por uno
- Verificar sintaxis después de cada cambio
- Commit individual para cada fix
