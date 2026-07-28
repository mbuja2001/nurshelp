with open(r'c:\Users\mbule\Videos\nursehelpdesk\nurshelp\frontend\src\components\dashboards\PhysicianDashboard.jsx', encoding='utf-8') as f:
    lines = f.readlines()

# Find first unclosed paren after line 1758
stack = []
for i in range(1757, 2065):
    line = lines[i]
    for j, char in enumerate(line):
        if char == '(':
            stack.append((i+1, j, char))  # line number, char position, char
        elif char == ')':
            if stack:
                opened_line, opened_pos, _ = stack.pop()
            else:
                print(f"Unmatched ) at line {i+1}, pos {j}")
                
if stack:
    print(f"\nUnclosed parens: {len(stack)}")
    for line_num, pos, char in stack:
        # Show the line
        context = lines[line_num-1][max(0, pos-20):min(len(lines[line_num-1]), pos+20)]
        print(f"Line {line_num}, pos {pos}: ...{repr(context)}...")
