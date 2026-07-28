import re

# Read the file
with open(r'c:\Users\mbule\Videos\nursehelpdesk\nurshelp\frontend\src\Dashboard\NurseDashboard.jsx', encoding='utf-8') as f:
    lines = f.readlines()

# Fix indentation for lines 1226-1679 (indices 1225-1678)
# Add 2 spaces to the beginning of each line that's not empty
for i in range(1225, 1679):  # lines 1226-1679 (0-indexed: 1225-1678)
    if lines[i].strip():  # if line is not empty
        lines[i] = '  ' + lines[i]  # add 2 spaces

# Write back
with open(r'c:\Users\mbule\Videos\nursehelpdesk\nurshelp\frontend\src\Dashboard\NurseDashboard.jsx', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Indentation fixed for fragment content (lines 1226-1679)")
