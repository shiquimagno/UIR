import re

# Read the file
with open('app_fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Remove light mode toggle (lines 715-765)
# Find and remove the entire section from "# Usuario y controles" to the second st.markdown
pattern1 = r'# Usuario y controles\s+st\.sidebar\.markdown\(f"👤 \*\*\{username\}\*\*"\).*?st\.sidebar\.markdown\("---"\)'
content = re.sub(pattern1, 'st.sidebar.markdown("---")', content, flags=re.DOTALL)

# Fix 2: Add "Again" logic to process_review
# Find the process_review function and add the logic after save_state
pattern2 = r'(def process_review\(card: Card, grade: int, session: dict\):.*?save_state\(state\))'
replacement2 = r'\1\n    \n    # If "Again" (grade=0), re-add card to end of queue\n    if grade == 0:\n        current_card_idx = session[\'cards_to_review\'][session[\'current_card_idx\']]\n        session[\'cards_to_review\'].append(current_card_idx)'

content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

# Write back
with open('app_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed app_fixed.py")
