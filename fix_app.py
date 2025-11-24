#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Read file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix 1: Remove lines 718-766 (user controls and theme CSS)
# But keep line 717 (st.sidebar.title) and line 767 (st.sidebar.markdown("---"))
# Lines to remove: 718-766 (the "# Usuario y controles" section through the theme CSS)
output = lines[:717] + lines[766:]

# Fix 2: Add "Again" logic after line that contains "save_state(state)" in process_review
final_output = []
for i, line in enumerate(output):
    final_output.append(line)
    
    # Look for save_state(state) in process_review function
    if 'save_state(state)' in line and i > 1300:
        # Check if we're in process_review by looking back
        context = ''.join(output[max(0, i-15):i])
        if 'def process_review' in context:
            # Add the Again logic
            final_output.append('    \n')
            final_output.append('    # Si es "Again" (grade=0), volver a agregar la tarjeta al final de la cola\n')
            final_output.append('    if grade == 0:\n')
            final_output.append('        current_card_idx = session[\'cards_to_review\'][session[\'current_card_idx\']]\n')
            final_output.append('        session[\'cards_to_review\'].append(current_card_idx)\n')

# Write fixed file
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(final_output)

print("Fixed! Removed light mode section and added Again logic")
print(f"Original: {len(lines)} lines")
print(f"Final: {len(final_output)} lines")
print(f"Removed: {len(lines) - len(final_output) + 5} lines")
