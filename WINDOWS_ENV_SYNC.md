# Windows Environment Sync Guide (Multi-PC Setup)

**Problem:** Colorama and dependency conflicts when working across 2 PCs with different Python/pip versions.

**Solution:** Pinned requirements + environment setup script

---

## 🎯 Quick Fix (Do This Now)

### Step 1: Sync Requirements (All PCs)
```powershell
# Windows PowerShell
cd C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML

# Delete old venv to reset
Remove-Item venv -Recurse -Force

# Create fresh venv
python -m venv venv

# Activate (you'll see prompt change to (venv))
.\venv\Scripts\Activate.ps1

# Install pinned versions (THIS IS THE KEY)
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 2: Verify Installation
```powershell
pip list | findstr colorama
pip list | findstr click
pip list | findstr werkzeug
```

You should see:
```
click                8.1.x or higher
colorama             0.4.6 or higher  
werkzeug             2.3.x or higher
```

---

## 📋 What Changed in requirements.txt

**Before:** Unpinned (causes version conflicts)
```
flask
pandas
requests
numpy
torch
sentence-transformers
llama-cpp-python
rank_bm25
tqdm
scipy
```

**After:** Pinned versions + explicit colorama/click
```
flask>=2.3.0
werkzeug>=2.3.0
colorama>=0.4.6      ← EXPLICIT (no more missing)
click>=8.1.0         ← EXPLICIT (prevents conflicts)
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
requests>=2.31.0
torch>=2.0.0
sentence-transformers>=2.2.0
llama-cpp-python>=0.2.0
rank_bm25>=0.2.2
tqdm>=4.65.0
python-dotenv>=1.0.0 ← NEW (needed for env vars)
```

---

## 🔄 Sync Process Across 2 PCs

### PC #1 (Source)
```powershell
# After running Step 1 & 2, create lockfile
pip freeze > requirements.lock
```

### PC #2 (Target)
```powershell
cd C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML

# Copy requirements.lock from PC #1
# Then:
Remove-Item venv -Recurse -Force
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install exact versions
pip install -r requirements.lock
```

---

## 🛠️ Setup Script (Automated)

Create this file: `setup-windows-env.ps1`

```powershell
# setup-windows-env.ps1
# Run: powershell -ExecutionPolicy Bypass -File setup-windows-env.ps1

Write-Host "🔧 Windows Environment Setup" -ForegroundColor Cyan
Write-Host ""

$ML_DIR = "C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML"
$VENV_DIR = "$ML_DIR\venv"

# Check Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Remove old venv
Write-Host "[2/5] Removing old environment..." -ForegroundColor Yellow
if (Test-Path $VENV_DIR) {
    Remove-Item $VENV_DIR -Recurse -Force
    Write-Host "  ✅ Cleaned"
}

# Create fresh venv
Write-Host "[3/5] Creating virtual environment..." -ForegroundColor Yellow
cd $ML_DIR
python -m venv venv
Write-Host "  ✅ Created"

# Activate and install
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
& $VENV_DIR\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Write-Host "  ✅ Installed"

# Verify
Write-Host "[5/5] Verifying Windows compatibility..." -ForegroundColor Yellow
pip list | findstr "colorama"
pip list | findstr "click"
pip list | findstr "werkzeug"

Write-Host ""
Write-Host "✅ Environment ready! Use: .\venv\Scripts\Activate.ps1" -ForegroundColor Green
```

**Run it:**
```powershell
powershell -ExecutionPolicy Bypass -File setup-windows-env.ps1
```

---

## 🚀 Troubleshooting Colorama Issues

### Issue: "ModuleNotFoundError: No module named 'colorama'"
```powershell
# Make sure venv is activated (prompt should show (venv))
.\venv\Scripts\Activate.ps1

# Then reinstall
pip install colorama>=0.4.6 --force-reinstall
```

### Issue: "click.exceptions.UsageError"
```powershell
# Click version mismatch
pip install click>=8.1.0 --upgrade
```

### Issue: "Flask banner errors on startup"
```powershell
# This is handled by ESI_Engine.py lines 9329+
# But make sure environment is clean:
pip uninstall colorama click werkzeug -y
pip install -r requirements.txt
```

### Issue: Different versions between 2 PCs
```powershell
# PC #1 - create lock file
pip freeze > requirements.lock

# PC #2 - sync exact versions
pip install -r requirements.lock
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] `pip list` shows colorama 0.4.6+
- [ ] `pip list` shows click 8.1.0+
- [ ] `pip list` shows werkzeug 2.3.0+
- [ ] ESI_Engine.py starts without "colorama" errors
- [ ] Frontend runs without terminal color issues
- [ ] Both PCs have identical `pip freeze` output

---

## 📌 For Future Multi-PC Work

Always:
1. Use `pip freeze > requirements.lock` after first PC setup
2. Use `pip install -r requirements.lock` on second PC
3. Never manually install packages (breaks sync)
4. If one PC needs a new package, update requirements.txt and sync both

This prevents the "I keep having to fix colorama" problem permanently.
