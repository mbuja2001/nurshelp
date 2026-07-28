with open(r'c:\Users\mbule\Videos\nursehelpdesk\nurshelp\frontend\src\components\dashboards\PhysicianDashboard.jsx', encoding='utf-8') as f:
    lines = f.readlines()

# Check queue section braces and parens (lines 1758-2065)
queue_section = ''.join(lines[1757:2065])  # 0-indexed

# Count all relevant brackets/parens
open_braces = queue_section.count('{')
close_braces = queue_section.count('}')
open_parens = queue_section.count('(')
close_parens = queue_section.count(')')
open_angles = queue_section.count('<')
close_angles = queue_section.count('>')

print(f"Braces: {open_braces} opens vs {close_braces} closes = {open_braces - close_braces}")
print(f"Parens: {open_parens} opens vs {close_parens} closes = {open_parens - close_parens}")
print(f"Angles: {open_angles} opens vs {close_angles} closes = {open_angles - close_angles}")

# Now do line-by-line to find the imbalance
paren_depth = 0
for i, line in enumerate(lines[1757:2065], start=1758):
    for char in line:
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
    if paren_depth < 0:
        print(f"ERROR: Negative paren depth at line {i}")
        break
    if i % 50 == 0 or paren_depth != 0:
        print(f"Line {i}: depth={paren_depth}")

print(f"Final paren depth: {paren_depth}")
