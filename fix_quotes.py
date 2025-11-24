import re

# Leer el archivo
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Procesar línea por línea
output = []
in_docstring = False
docstring_start = None

for i, line in enumerate(lines, 1):
    # Detectar inicio/fin de docstring
    if '"""' in line:
        if not in_docstring:
            # Inicio de docstring - convertir a comentario
            docstring_start = i
            in_docstring = True
            # Reemplazar """ con #
            new_line = line.replace('"""', '#')
            output.append(new_line)
        else:
            # Fin de docstring - skip esta línea
            in_docstring = False
            continue
    elif in_docstring:
        # Dentro de docstring - convertir a comentario
        if line.strip():
            output.append('    # ' + line.strip() + '\n')
    else:
        # Línea normal
        output.append(line)

# Guardar
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(output)

print("Fixed all docstrings")
