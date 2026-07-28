import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout)

content = open(r"c:\Users\mbule\AppData\Roaming\Code\copilot-terminal-output\copilot-terminal-output-02b97d4d-2d56-45de-9750-a793d420d203.txt", encoding='utf-8', errors='ignore').read()
lines = content.split('\n')
# Find lines where balance is 1 or 2 (meaning it should be closing)
print("Lines where balance IS 1 or 2:")
for line in lines:
    if 'cumul_balance=1 |' in line or 'cumul_balance=2 |' in line:
        # Extract just the line number and first part, avoiding emojis
        parts = line.split('|')
        if len(parts) >= 2:
            print(f"{parts[0]}| {parts[1][:50]}...")
