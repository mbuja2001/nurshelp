# ⚡ QUICK FIX: Colorama Conflicts (2-PC Setup)

## 🎯 One-Time Setup (Both PCs)

### PC #1 (First Time)
```powershell
cd C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML
powershell -ExecutionPolicy Bypass -File setup-windows-env.ps1
```

### PC #2 (Sync from PC #1)
```powershell
# 1. Copy requirements.lock from PC #1
# 2. Run on PC #2:
cd C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML
powershell -ExecutionPolicy Bypass -File sync-environments.ps1 -Mode apply-lock
```

---

## 🔄 After Adding New Packages

**If you install a new package on one PC:**

```powershell
# On that PC, update the lock:
pip freeze > requirements.lock

# Copy requirements.lock to other PC
# On other PC:
powershell -ExecutionPolicy Bypass -File sync-environments.ps1 -Mode apply-lock
```

---

## ✅ Verify It Works

```powershell
.\venv\Scripts\Activate.ps1
pip list | findstr colorama
```

Should show: `colorama              0.4.6` (or higher)

---

## 🚨 If Colorama Still Breaks

```powershell
# Nuclear option - complete reset:
Remove-Item venv -Recurse -Force
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.lock
```

---

## 📌 Remember

| Issue | Solution |
|-------|----------|
| "colorama error" | Run setup script |
| "click error" | Run setup script |
| Different versions on 2 PCs | Use `sync-environments.ps1 -Mode apply-lock` |
| Need to add new package | Update requirements.txt → both PCs rebuild |

---

## 📂 Files Created

- `requirements.txt` - Pinned versions (updated)
- `requirements.lock` - Exact locked versions (create after setup)
- `setup-windows-env.ps1` - Initial setup script
- `sync-environments.ps1` - PC-to-PC sync script
- `WINDOWS_ENV_SYNC.md` - Full documentation

---

**TL;DR:** Run the setup script once per PC. Never manually pip install again. Use sync script when adding packages. Done! ✅
