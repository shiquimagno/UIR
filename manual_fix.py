import sys

# Read original file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix 1: Remove lines 717-765 (light mode toggle)
# Keep only sidebar title and separator
output_lines = []
skip_mode = False
line_num = 0

for i, line in enumerate(lines, 1):
    line_num = i
    
    # Start skipping at line 717
    if i == 717 and '# Usuario y controles' in line:
        skip_mode = True
        continue
    
    # Stop skipping at line 766 (after the else block)
    if i == 766 and skip_mode:
        skip_mode = False
        continue
    
    if not skip_mode:
        output_lines.append(line)

# Fix 2: Add "Again" logic after save_state in process_review
# Find the line with save_state(state) in process_review function
final_lines = []
for i, line in enumerate(output_lines):
    final_lines.append(line)
    
    # After save_state(state) in process_review, add the Again logic
    if 'save_state(state)' in line and i > 1300:  # Approximate line number
        # Check if we're in process_review function
        context = ''.join(output_lines[max(0, i-20):i])
        if 'def process_review' in context:
            final_lines.append('    \n')
            final_lines.append('    # If "Again" (grade=0), re-add card to end of queue\n')
            final_lines.append('    if grade == 0:\n')
            final_lines.append('        current_card_idx = session[\'cards_to_review\'][session[\'current_card_idx\']]\n')
            final_lines.append('        session[\'cards_to_review\'].append(current_card_idx)\n')

# Write fixed version
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(final_lines)

print(f"Fixed! Removed {len(lines) - len(final_lines) + 5} lines, added 5 lines for Again logic")
