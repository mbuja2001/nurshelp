# 🚀 QUICK START: Fix Cross-Platform Issues

## What Was Found

Your system has **50+ hardcoded Linux paths** that prevent Windows execution:
- `/home/wtc/Videos/nursehelpdesk/...`
- `/home/wtc/Documents/living_compute_labs/...`

Plus **colorama/click conflicts** that cause Flask console spam.

---

## 3-Step Fix

### Step 1: Apply Critical ESI_Engine.py Fix (5 minutes)

**Replace this code** in [ESI_Engine.py](nurshelp/backend/ML/ESI_Engine.py#L320) (around line 315-340):

```python
# ❌ OLD CODE - Delete this section
_GGUF_CANDIDATES = [
    os.path.join(script_dir, "models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf"),
    os.environ.get("GGUF_MODEL_PATH"),
    "/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf",
    "/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf",
]
```

**With this code** (copy/paste):

```python
# ✅ NEW CODE - Universal cross-platform path detection
def _get_gguf_model_path():
    """Cross-platform GGUF model detection with fallback chain."""
    import platform
    
    script_dir = Path(__file__).resolve().parent
    
    # Priority 1: Environment variable (highest priority for deployment)
    env_path = os.environ.get("GGUF_MODEL_PATH")
    if env_path and Path(env_path).exists():
        print(f"[startup] ✅ GGUF from env variable: {env_path}")
        return env_path
    
    # Priority 2: Relative to script (works on all platforms in dev)
    relative_candidates = [
        script_dir / "models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf",
        script_dir / "../models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf",
        script_dir / "../../models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf",
    ]
    
    for candidate in relative_candidates:
        if candidate.exists():
            print(f"[startup] ✅ GGUF from relative path: {candidate}")
            return str(candidate)
    
    # Priority 3: Platform-specific absolute paths
    system = platform.system()
    
    if system == "Windows":
        windows_candidates = [
            Path("C:/Users/mbule/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf"),
            Path(os.path.expanduser("~/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf")),
        ]
        for candidate in windows_candidates:
            if candidate.exists():
                print(f"[startup] ✅ GGUF from Windows path: {candidate}")
                return str(candidate)
    
    elif system == "Linux" or system == "Darwin":
        unix_candidates = [
            Path("/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf"),
            Path("/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf"),
            Path(os.path.expanduser("~/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf")),
        ]
        for candidate in unix_candidates:
            if candidate.exists():
                print(f"[startup] ✅ GGUF from {system} path: {candidate}")
                return str(candidate)
    
    # Fallback with informative error
    default = str(script_dir / "models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf")
    print(f"[startup] ⚠️  GGUF model not found. Set GGUF_MODEL_PATH environment variable.")
    print(f"[startup]     Expected: {default}")
    return default

GGUF_MODEL_PATH = _get_gguf_model_path()
```

### Step 2: Set Environment Variable

**On Windows:**
```powershell
# Temporary (current session only)
$env:GGUF_MODEL_PATH = "C:\Users\mbule\Videos\nursehelpdesk\backend\ML\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\medgemma-4b-it-Q4_K_M.gguf"

# Permanent (all future sessions)
[System.Environment]::SetEnvironmentVariable("GGUF_MODEL_PATH", "C:\Users\mbule\Videos\nursehelpdesk\backend\ML\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\medgemma-4b-it-Q4_K_M.gguf", "User")
```

**On Linux/Mac:**
```bash
# Temporary
export GGUF_MODEL_PATH="/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export GGUF_MODEL_PATH="/home/wtc/Videos/..."' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Test It Works

```bash
# Start server
python ESI_Engine.py

# Check logs - should show:
# [startup] ✅ GGUF from ...

# Test endpoint
curl -X POST http://localhost:5000/llm-triage \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "TEST123", "vitals": {...}}'
```

---

## File-by-File Fixes (Optional - Use Fixer Script)

Run the automated fixer script:

```bash
cd c:\Users\mbule\Videos\nursehelpdesk
python fix_cross_platform.py
```

This automatically fixes:
- ✅ download_medgemma_model.py
- ✅ download_llama_model.py  
- ✅ 10+ other download scripts
- ✅ inspect_pmc_parquet.py
- ✅ inspect_faiss_data.py

---

## Verification Checklist

After applying fixes, verify:

- [ ] ESI_Engine.py contains `_get_gguf_model_path()` function
- [ ] GGUF_MODEL_PATH environment variable is set
- [ ] Server starts without "model not found" errors
- [ ] Logs show `✅ GGUF from ...` at startup
- [ ] `/llm-triage` endpoint responds to test requests
- [ ] No colorama/click console warnings

---

## Common Issues & Solutions

### Issue: "GGUF model not found"
**Solution:** Set GGUF_MODEL_PATH environment variable to actual file location
```bash
# Find the actual file first
find c:\Users\mbule\Videos -name "medgemma-4b-it-Q4_K_M*.gguf" 2>/dev/null
# Then set that path as GGUF_MODEL_PATH
```

### Issue: "colorama" or "click" errors
**Solution:** Already fixed in updated ESI_Engine.py (lines 9329-9350)

### Issue: Works on Linux but still fails on Windows  
**Solution:** Check that `_get_gguf_model_path()` is actually called. Look for:
```python
GGUF_MODEL_PATH = _get_gguf_model_path()  # Should be present
```

### Issue: "Cannot import from diagnostic_criteria"
**Solution:** This is expected on first run. The system gracefully falls back to deterministic mode.

---

## Documentation

For complete details, see: [CROSS_PLATFORM_COMPATIBILITY_FIXES.md](CROSS_PLATFORM_COMPATIBILITY_FIXES.md)

---

## Support Commands

### Windows PowerShell
```powershell
# Check environment variable is set
echo $env:GGUF_MODEL_PATH

# Check file exists
Test-Path $env:GGUF_MODEL_PATH

# Run server with debug
python -u ESI_Engine.py

# Kill stuck processes
Stop-Process -Name python -Force
```

### Linux/Mac
```bash
# Check environment variable
echo $GGUF_MODEL_PATH

# Check file exists  
test -f $GGUF_MODEL_PATH && echo "File exists" || echo "Not found"

# Run with debug output
python3 -u ESI_Engine.py

# Find running processes
ps aux | grep ESI_Engine
```

---

**Status:** ✅ Ready to apply | **Time to fix:** ~15 minutes | **Risk:** Low (backward compatible)
