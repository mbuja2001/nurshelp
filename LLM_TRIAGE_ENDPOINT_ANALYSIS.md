# LLM Triage Endpoint Analysis: Cross-Platform Loading Issues

## Executive Summary

Your ESI_Engine.py has **critical cross-platform incompatibilities** in:
1. **Model path handling** - hardcoded Linux path breaks on Windows
2. **GGUF initialization** - thread-safe locking differs between systems
3. **Environment variable setup** - cache directories not cross-platform
4. **Hardware detection** - CPU detection broken on Windows

**Status**: ❌ Windows broken | ✅ Linux working

---

## 1. THE SMOKING GUN: Hardcoded Linux Path (Line ~600)

### Current Code (BROKEN on Windows):
```python
GGUF_MODEL_PATH = "/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf"
```

### Problem:
- ❌ `/home/wtc/` - Linux home directory path
- ❌ Forward slashes `/` - won't resolve on Windows
- ❌ No drive letter (C:\, D:\, etc.) - invalid on Windows

### Why Linux Works:
- The Linux machine has the file at exactly that path
- When code loads, it finds the file and initializes GGUF successfully
- All subsequent inference calls work

### Why Windows Fails:
```
FileNotFoundError: /home/wtc/Videos/...
  (this path doesn't exist on Windows at all)
```

---

## 2. LLM TRIAGE ENDPOINT FLOW (When Being Used)

### Current Flow (Line 3825+):

```
POST /llm-triage
    ↓
[Step 1: REQUEST DEDUPLICATION]
    - Checks if patient was already triaged in last 15s
    - Prevents duplicate submissions from browser
    ↓
[Step 2: HARDWARE DETECTION]
    - Detects GPU (torch.cuda.is_available())
    - Checks GGUF availability (_GGUF_AVAILABLE)
    - Routes to compute_mode: "llm_rag" | "llm_only" | "deterministic"
    ↓
[Step 3: VITALS VALIDATION & TEWS]
    - Validates incoming vitals (sanity checks)
    - Calculates TEWS score (South African Track & Trigger)
    - Applies validation_flags (e.g., safety overrides)
    ↓
[Step 4: SAFETY GATES]
    - Vital sign triggers (hard rules)
    - Red flag keywords (deterministic)
    - Semantic red flags (pattern matching)
    ↓
[Step 5: RAG RETRIEVAL] (if available)
    - Calls RAG_RETRIEVER.retrieve()
    - Gets top 10 similar cases from PubMedBERT
    ↓
[Step 6: CALL GGUF_GENERATE()]  ← THIS IS WHERE IT FAILS ON WINDOWS
    - prompt = triage_prompt (clinical triage decision)
    - max_tokens = 4000
    - temperature = 0.0
    - Calls: gguf_generate(prompt, max_tokens=4000, temperature=0.0)
    ↓
[Step 7: JSON PARSING]
    - Extracts 14-field JSON from GGUF output
    - Maps to supervisor_output dict
    - Handles truncated/corrupted JSON gracefully
    ↓
[Step 8: BUILD RESPONSE]
    - Merges all clinical data
    - Returns JSON response compatible with frontend
```

### Where Linux Works vs Windows Fails:

| Step | Linux | Windows |
|------|-------|---------|
| Step 1-5 | ✅ Works | ✅ Works (no system-dependent code) |
| Step 6 | ✅ **GGUF loads from `/home/wtc/...` and runs** | ❌ **Path doesn't exist → FileNotFoundError** |
| Step 7-8 | ✅ (if step 6 succeeds) | ❌ (crashes at step 6) |

---

## 3. MODEL LOADING DIFFERENCES: Windows vs Linux

### A. Path Resolution Issues

#### Linux (Working):
```python
# Line ~600
GGUF_MODEL_PATH = "/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf"

# Resolves to:
# /home/wtc/Videos/nursehelpdesk/... ✅ (valid on Linux)
```

#### Windows (Broken):
```python
# Same path attempted on Windows:
# /home/wtc/Videos/... ❌ (invalid - no /home on Windows)

# Expected:
# C:\Users\mbule\Videos\nursehelpdesk\... ✅ (valid on Windows)
```

### B. Forward Slashes vs Backslashes

#### Python's Behavior:
```python
# Python accepts BOTH on Windows, but:
path_fwd = "/home/wtc/file.gguf"  # Works on Linux ONLY
path_back = "C:\\Users\\file.gguf"  # Works on Windows (double backslash)
path_fwd_win = "C:/Users/file.gguf"  # Also works on Windows! (Python auto-converts)

# But! Leading slash on Windows is interpreted as "root of current drive"
# So "/home/wtc/file.gguf" on Windows C:\ becomes "C:\home\wtc\file.gguf" ❌
```

### C. GGUF Library Behavior

#### load_gguf_model() (Line ~650):
```python
def load_gguf_model():
    from llama_cpp import Llama
    
    # THIS LINE FAILS ON WINDOWS:
    if not os.path.exists(GGUF_MODEL_PATH):  # ← Checks hardcoded path
        print(f"GGUF file not found: {GGUF_MODEL_PATH}")
        return False  # Returns False, _GGUF_AVAILABLE = False
    
    # If file not found, entire LLM initialization fails
    _llm = Llama(model_path=GGUF_MODEL_PATH, ...)
```

#### On Windows at startup:
```
[INIT] ⚠️  GGUF file not found: /home/wtc/Videos/.../medgemma-4b-it-Q4_K_M-003.gguf
[INIT]    Download: huggingface-cli download ...
[startup] ⚠️  GGUF not available: falling back to deterministic rules only
```

---

## 4. ENVIRONMENT VARIABLE SETUP (Lines ~50-70)

### Current Code (Works on Linux ONLY):
```python
# Line ~50-70
_base_dir = Path(__file__).resolve().parent
_cache_dir = _base_dir / "model_cache"

# Sets env vars pointing to _cache_dir:
os.environ["TRANSFORMERS_CACHE"] = str(_cache_dir)
os.environ["HF_HOME"] = str(_cache_dir)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_cache_dir)

# On Linux: /home/wtc/.../backend/ML/model_cache ✅
# On Windows: C:\Users\mbule\...\backend\ML\model_cache ✅ (actually works!)
```

### Why This Part IS Cross-Platform Compatible:
```python
Path(__file__).resolve().parent  # Uses pathlib - cross-platform!
# Returns:
#   Linux:   PosixPath('/home/wtc/.../backend/ML')
#   Windows: WindowsPath('C:\\Users\\mbule\\...\\backend\\ML')
```

✅ **This part is OK** - pathlib handles both systems

---

## 5. HARDWARE DETECTION (Lines ~450-550)

### Current Code (PARTIALLY BROKEN on Windows):

```python
# Line ~500 - CPU Model Detection
try:
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
    cpu_model = cpu_info.get('brand_raw', 'Unknown CPU')
except:
    # Fallback: try to read from /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:  # ← LINUX ONLY!
            for line in f:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
            else:
                cpu_model = 'Unknown CPU'
    except:
        cpu_model = 'Unknown CPU'
```

### Problems on Windows:
```python
# /proc/cpuinfo doesn't exist on Windows!
# So this fails silently and falls back to 'Unknown CPU'

# But it STILL works because:
# 1. cpuinfo library (if installed) handles Windows correctly
# 2. Fallback to 'Unknown CPU' is safe
```

✅ **This part works by accident** (fallback is safe)

---

## 6. THREAD-SAFE GGUF INITIALIZATION (Lines ~630-650)

### Current Code:
```python
_gguf_lock = threading.Lock()  # ← Cross-platform! ✅

def gguf_generate(prompt, max_tokens=150, temperature=0.0):
    if not _GGUF_AVAILABLE or _llm is None:
        raise Exception("GGUF model not available")
    
    try:
        with _gguf_lock:  # Thread-safe lock ✅
            gc.collect()
            response = _llm(prompt=prompt, max_tokens=max_tokens, ...)
            return response['choices'][0]['text'].strip()
    except Exception as e:
        raise Exception(f"GGUF inference error: {e}")
```

✅ **Thread safety is cross-platform** - threading.Lock() works on both

---

## 7. SOLUTION: Cross-Platform Model Path Fix

### Option 1: Use Environment Variables (BEST for deployment)

```python
# At startup (after imports, before load_gguf_model):
import os
from pathlib import Path

def get_gguf_model_path():
    """Get GGUF model path, respecting platform differences."""
    
    # Try environment variable first (allows override)
    if "GGUF_MODEL_PATH" in os.environ:
        path = os.environ["GGUF_MODEL_PATH"]
        if os.path.exists(path):
            return path
        print(f"⚠️  GGUF_MODEL_PATH env var set but path not found: {path}")
    
    # Try common locations
    candidates = [
        # Windows (local development)
        Path("C:/Users/mbule/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf"),
        
        # Linux (production)
        Path("/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf"),
        
        # Relative to script (works on both)
        Path(__file__).parent / "models" / "medgemma-4b-it-Q4_K_M-003.gguf",
        
        # Docker/container mounted path
        Path("/models/medgemma-4b-it-Q4_K_M-003.gguf"),
    ]
    
    for candidate in candidates:
        if candidate.exists():
            print(f"✅ Found GGUF model at: {candidate}")
            return str(candidate)
    
    print(f"❌ GGUF model not found in any candidate location:")
    for c in candidates:
        print(f"   - {c}")
    return None

# Usage:
GGUF_MODEL_PATH = get_gguf_model_path()
```

### Option 2: Docker/Container Path (BEST for production)

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Copy model into container at standard location
COPY backend/ML/models/medgemma-4b-local/...medgemma-4b-it-Q4_K_M-003.gguf \
     /models/medgemma-4b-it-Q4_K_M-003.gguf

# Environment variable
ENV GGUF_MODEL_PATH=/models/medgemma-4b-it-Q4_K_M-003.gguf

# Then Python finds it at /models/... (works on any system inside container)
```

### Option 3: Use download_model() on First Run (BEST for users)

```python
def download_model_if_missing():
    """Download GGUF model if not found locally."""
    from huggingface_hub import hf_hub_download
    
    cache_dir = Path(__file__).parent / "model_cache"
    cache_dir.mkdir(exist_ok=True)
    
    model_name = "medgemma-4b-it-Q4_K_M-003.gguf"
    local_path = cache_dir / model_name
    
    if local_path.exists():
        print(f"✅ Model found at {local_path}")
        return str(local_path)
    
    print(f"📥 Downloading {model_name}...")
    model_path = hf_hub_download(
        repo_id="unsloth/medgemma-4b-it-GGUF",
        filename=model_name,
        cache_dir=str(cache_dir),
        force_download=False
    )
    print(f"✅ Model downloaded to {model_path}")
    return model_path

# Usage:
GGUF_MODEL_PATH = download_model_if_missing()
```

---

## 8. COMPLETE CROSS-PLATFORM FIX

### Replace Lines 600-610 with:

```python
# ===== CROSS-PLATFORM GGUF MODEL PATH DETECTION =====
def get_gguf_model_path():
    """Detect GGUF model path across Windows/Linux/Docker."""
    
    # Priority 1: Environment variable
    if "GGUF_MODEL_PATH" in os.environ:
        env_path = os.environ["GGUF_MODEL_PATH"]
        if os.path.exists(env_path):
            print(f"[GGUF] Using GGUF_MODEL_PATH from environment: {env_path}")
            return env_path
        print(f"[GGUF] ⚠️  GGUF_MODEL_PATH set but not found: {env_path}")
    
    # Priority 2: Relative to script (works on both Windows/Linux)
    script_dir = Path(__file__).resolve().parent
    relative_locations = [
        script_dir / "models" / "medgemma-4b-it-Q4_K_M-003.gguf",
        script_dir / "../models/medgemma-4b-it-Q4_K_M-003.gguf",
        script_dir / "../../models/medgemma-4b-it-Q4_K_M-003.gguf",
    ]
    
    for path in relative_locations:
        if path.exists():
            print(f"[GGUF] Found model relative to script: {path}")
            return str(path)
    
    # Priority 3: Absolute paths (one will match on current platform)
    platform_paths = {
        'Windows': "C:/Users/mbule/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf",
        'Linux': "/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf",
        'Docker': "/models/medgemma-4b-it-Q4_K_M-003.gguf",
    }
    
    for platform_name, path in platform_paths.items():
        if os.path.exists(path):
            print(f"[GGUF] Found model at {platform_name} location: {path}")
            return path
    
    # Priority 4: Try to download from HuggingFace
    print("[GGUF] Model not found locally, attempting download...")
    try:
        from huggingface_hub import hf_hub_download
        cache_dir = script_dir / "model_cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        model_path = hf_hub_download(
            repo_id="unsloth/medgemma-4b-it-GGUF",
            filename="medgemma-4b-it-Q4_K_M-003.gguf",
            cache_dir=str(cache_dir)
        )
        print(f"[GGUF] ✅ Downloaded model to: {model_path}")
        return model_path
    except Exception as dl_error:
        print(f"[GGUF] ⚠️  Download failed: {dl_error}")
    
    # All options exhausted
    print("[GGUF] ❌ GGUF model path not found and download failed")
    print("[GGUF]    Options:")
    print("[GGUF]    1. Set GGUF_MODEL_PATH environment variable")
    print("[GGUF]    2. Place model at any relative location above")
    print("[GGUF]    3. Install huggingface_hub: pip install huggingface_hub")
    return None

# Use it:
GGUF_MODEL_PATH = get_gguf_model_path()
if not GGUF_MODEL_PATH:
    print("[startup] ⚠️  GGUF model unavailable - LLM inference disabled")
    _GGUF_AVAILABLE = False
    _llm = None
else:
    print(f"[startup] Loading GGUF model: {GGUF_MODEL_PATH}")
    # Continue with load_gguf_model()
```

---

## 9. ENVIRONMENT SETUP FOR CROSS-PLATFORM

### .env File (add to project root):
```bash
# Windows
GGUF_MODEL_PATH=C:/Users/mbule/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf
BACKEND_PORT=5000

# Or Linux
# GGUF_MODEL_PATH=/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf
```

### startup.py (load before Flask app):
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Loads .env file

GGUF_MODEL_PATH = os.environ.get("GGUF_MODEL_PATH")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 5000))
```

---

## 10. TESTING THE FIX

### Test on Windows:
```powershell
# 1. Set environment variable
$env:GGUF_MODEL_PATH="C:/Users/mbule/Videos/nursehelpdesk/backend/ML/models/.../medgemma-4b-it-Q4_K_M-003.gguf"

# 2. Start server
python ESI_Engine.py

# 3. Check logs
# Should see: [GGUF] Found model at Windows location: C:/Users/mbule/...
```

### Test on Linux:
```bash
# 1. Set environment variable
export GGUF_MODEL_PATH="/home/wtc/Videos/nursehelpdesk/backend/ML/models/.../medgemma-4b-it-Q4_K_M-003.gguf"

# 2. Start server
python3 ESI_Engine.py

# 3. Check logs
# Should see: [GGUF] Found model at Linux location: /home/wtc/...
```

### Test in Docker:
```bash
# Dockerfile includes:
ENV GGUF_MODEL_PATH=/models/medgemma-4b-it-Q4_K_M-003.gguf

# When container starts, Python finds it at /models/...
```

---

## 11. SUMMARY TABLE

| Issue | Linux | Windows | Fix |
|-------|-------|---------|-----|
| **Path hardcoded** | ✅ Works | ❌ Fails | Use `get_gguf_model_path()` |
| **Forward/backslashes** | ✅ Works | ⚠️ Fragile | Use `pathlib.Path` |
| **Cache dir detection** | ✅ Works | ✅ Works | Already uses `pathlib` |
| **CPU detection** | ✅ Works | ⚠️ Fallback | Works via fallback |
| **Thread safety** | ✅ Works | ✅ Works | `threading.Lock()` is universal |
| **GGUF inference** | ✅ Works | ❌ If path missing | Same thread-safe code |

---

## 12. RECOMMENDED DEPLOYMENT

### For Development:
```bash
# Set env var for your system
export GGUF_MODEL_PATH="/path/to/your/model.gguf"  # Linux
set GGUF_MODEL_PATH=C:/path/to/your/model.gguf     # Windows
python ESI_Engine.py
```

### For Production:
```dockerfile
# Docker is system-agnostic - use container paths
ENV GGUF_MODEL_PATH=/models/medgemma-4b-it-Q4_K_M-003.gguf
```

### For CI/CD (GitHub Actions):
```yaml
# .github/workflows/test.yml
steps:
  - run: |
      export GGUF_MODEL_PATH="/tmp/models/medgemma.gguf"
      python -m pytest tests/
```

---

## Implementation Checklist

- [ ] Replace hardcoded GGUF path with `get_gguf_model_path()` function
- [ ] Add environment variable support
- [ ] Test on Windows with GGUF_MODEL_PATH env var set
- [ ] Test on Linux with GGUF_MODEL_PATH env var set
- [ ] Test with missing model (should gracefully fall back)
- [ ] Add download_model_if_missing() fallback
- [ ] Create .env.example file for users
- [ ] Document in README: "Windows users must set GGUF_MODEL_PATH"
- [ ] Update Docker deployment with ENV GGUF_MODEL_PATH
- [ ] Test /llm-triage endpoint on both platforms
