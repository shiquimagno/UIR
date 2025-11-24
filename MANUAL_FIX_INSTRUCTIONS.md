# Manual Fix Instructions for app.py

## Problem
Automatic edits keep breaking the file with indentation errors. Manual editing required.

## Fix 1: Remove Light Mode (Lines 717-765)

**DELETE these lines:**
```
Lines 717-730: All the "Usuario y controles" section
Lines 721-726: Toggle button code  
Lines 728-730: Logout button
Lines 732-765: All the theme CSS (both dark and light)
```

**KEEP only:**
```python
st.sidebar.title("🧠 Simulador UIR/UIC")
st.sidebar.markdown("---")

pages = [
```

## Fix 2: Add "Again" Logic (Around line 1372)

**FIND this code:**
```python
    # Guardar
    save_state(state)
    
    # Avanzar (Streamlit hará rerun automáticamente después del callback)
    session['current_card_idx'] += 1
```

**REPLACE with:**
```python
    # Guardar
    save_state(state)
    
    # Si es "Again" (grade=0), volver a agregar la tarjeta al final de la cola
    if grade == 0:
        current_card_idx = session['cards_to_review'][session['current_card_idx']]
        session['cards_to_review'].append(current_card_idx)
    
    # Avanzar (Streamlit hará rerun automáticamente después del callback)
    session['current_card_idx'] += 1
```

## After Manual Edits

Run:
```bash
python -m py_compile app.py
git add app.py
git commit -m "Remove light mode and fix Again to re-add cards"
git push
```
