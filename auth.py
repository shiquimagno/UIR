"""
Sistema de autenticación para Spaced Repetition Simulator
Usa archivos JSON para almacenar usuarios (sin necesidad de MySQL)
"""

import json
import os
import hashlib
import streamlit as st
from datetime import datetime
from typing import Optional, Dict

USERS_FILE = "data/users.json"

def hash_password(password: str) -> str:
    """Hash de contraseña usando SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> Dict[str, dict]:
    """Cargar usuarios desde JSON"""
    if not os.path.exists(USERS_FILE):
        return {}
    
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_users(users: Dict[str, dict]):
    """Guardar usuarios a JSON"""
    os.makedirs("data", exist_ok=True)
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    Registrar nuevo usuario
    
    Returns:
        (success, message)
    """
    if not username or not password:
        return False, "Usuario y contraseña son requeridos"
    
    if len(username) < 3:
        return False, "El usuario debe tener al menos 3 caracteres"
    
    if len(password) < 4:
        return False, "La contraseña debe tener al menos 4 caracteres"
    
    users = load_users()
    
    if username in users:
        return False, "El usuario ya existe"
    
    # Crear usuario
    users[username] = {
        'password_hash': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    
    save_users(users)
    
    # Crear archivo de estado para el usuario
    user_state_file = f"data/user_{username}_state.json"
    if not os.path.exists(user_state_file):
        with open(user_state_file, 'w', encoding='utf-8') as f:
            json.dump({
                'cards': [],
                'params': {
                    'alpha': 0.2,
                    'gamma': 0.15,
                    'delta': 0.02,
                    'eta': 0.05
                },
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
    
    return True, "Usuario registrado exitosamente"

def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Autenticar usuario
    
    Returns:
        (success, message)
    """
    if not username or not password:
        return False, "Usuario y contraseña son requeridos"
    
    users = load_users()
    
    if username not in users:
        return False, "Usuario no existe"
    
    if users[username]['password_hash'] != hash_password(password):
        return False, "Contraseña incorrecta"
    
    # Actualizar último login
    users[username]['last_login'] = datetime.now().isoformat()
    save_users(users)
    
    return True, "Login exitoso"

def show_auth_page():
    """
    Mostrar página de autenticación (login/registro)
    """
    st.title("🔐 Spaced Repetition Simulator")
    st.markdown("### Sistema de Repaso Espaciado con UIR/UIC")
    
    tab1, tab2 = st.tabs(["🔑 Iniciar Sesión", "📝 Registrarse"])
    
    with tab1:
        st.subheader("Iniciar Sesión")
        
        with st.form("login_form"):
            username = st.text_input("Usuario", key="login_username")
            password = st.text_input("Contraseña", type="password", key="login_password")
            submit = st.form_submit_button("Entrar", use_container_width=True, type="primary")
            
            if submit:
                success, message = login_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    with tab2:
        st.subheader("Crear Cuenta Nueva")
        
        with st.form("register_form"):
            new_username = st.text_input("Usuario (mín. 3 caracteres)", key="reg_username")
            new_password = st.text_input("Contraseña (mín. 4 caracteres)", type="password", key="reg_password")
            confirm_password = st.text_input("Confirmar Contraseña", type="password", key="reg_confirm")
            submit_reg = st.form_submit_button("Registrarse", use_container_width=True, type="primary")
            
            if submit_reg:
                if new_password != confirm_password:
                    st.error("Las contraseñas no coinciden")
                else:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(message)
                        st.info("Ahora puedes iniciar sesión con tu cuenta")
                    else:
                        st.error(message)
    
    # Info adicional
    st.markdown("---")
    st.info("""
    **Características:**
    - 🧠 Algoritmo UIR/UIC personalizado
    - 📊 Analytics y estadísticas detalladas
    - 🔥 Sistema de rachas
    - 📈 Múltiples modos de repaso
    - 🌙 Modo oscuro/claro
    """)

def logout():
    """Cerrar sesión"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

def get_user_state_file(username: str) -> str:
    """Obtener archivo de estado para un usuario"""
    return f"data/user_{username}_state.json"
