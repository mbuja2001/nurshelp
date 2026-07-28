from pathlib import Path

p = Path(r'c:\Users\mbule\Videos\nursehelpdesk\nurshelp\frontend\src\components\dashboards\PhysicianDashboard.jsx')
lines = p.read_text(encoding='utf-8').splitlines()

# Do a line-by-line cumulative count to find where imbalance starts
print("Line-by-line cumulative brace balance (showing lines with braces):")
cumulative_opens = 0
cumulative_closes = 0
last_balanced_line = 1757

for i in range(1757, 2064):
    line = lines[i]
    opens = line.count('{')
    closes = line.count('}')
    cumulative_opens += opens
    cumulative_closes += closes
    balance = cumulative_opens - cumulative_closes
    
    if balance != 0 or opens > 0 or closes > 0:
        print(f"{i+1:4}: opens={opens}, closes={closes}, cumul_balance={balance} | {line.strip()[:70]}")
        if balance == 0:
            last_balanced_line = i

print(f"\nLast balanced line: {last_balanced_line + 1}")
print(f"Final cumulative balance: {cumulative_opens - cumulative_closes}")
