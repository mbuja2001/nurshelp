# 🗺️ SYSTEM CONFLICTS - VISUAL GUIDE

## Architecture: The Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                     NURSEHELP BACKEND                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼────┐        ┌─────▼────┐
              │ ESI_ENGINE│        │ML SCRIPTS│
              │  Python   │        │ 20 files │
              └─────┬────┘        └─────┬────┘
                    │                   │
        ╔═══════════▼═══════════╗   ╔═══▼═══════════╗
        ║  HARDCODED PATHS      ║   ║ HARDCODED     ║
        ║ /home/wtc/Videos/...  ║   ║ PATHS         ║
        ║ /home/wtc/Documents/..║   ║ /home/wtc/... ║
        ║                       ║   ║               ║
        ║   ❌ LINUX ONLY       ║   ║ ❌ LINUX ONLY ║
        ╚═══════════╤═══════════╝   ╚═══╤═══════════╝
                    │                   │
        ┌───────────┴───────────────────┘
        │
        ▼
    ┌──────────────────────────┐
    │   WINDOWS EXECUTION      │
    │   ❌ CRASH:              │
    │   FileNotFoundError      │
    │   /home/wtc/... DNE      │
    └──────────────────────────┘
```

---

## The Fix: Universal Path Detection

```
┌──────────────────────────────────────────────────────────────────┐
│                  _get_gguf_model_path()                          │
│                (New Universal Function)                          │
└──────────────────────────────────────────────────────────────────┘
              │
              ├─────────────────────────────────────┐
              │                                     │
        ┌─────▼──────┐                        ┌────▼─────┐
        │ PRIORITY 1  │                        │ PRIORITY │
        │Environment  │◄──────YES──────┐      │    2     │
        │Variable     │      EXISTS?    │      │Relative  │
        │GGUF_MODEL_  │                │      │Paths     │
        │PATH         │                │      └────┬─────┘
        └─────┬──────┘                │            │
              │                       │      ┌─────▼──────┐
              │                       └─────►│SUCCESS     │
              │   NO                         │RETURN PATH │
              │                             └────────────┘
              │
        ┌─────▼──────────┐
        │ PRIORITY 3     │
        │Platform-Specific
        │Absolute Paths  │
        │                │
        │ Windows:       │
        │ C:/Users/...   │
        │                │
        │ Linux:         │
        │ /home/wtc/...  │
        │                │
        │ Mac:           │
        │ ~/Videos/...   │
        └─────┬──────────┘
              │
        ┌─────▼──────┐
        │ PRIORITY 4 │
        │Auto-Download
        │HuggingFace │
        └─────┬──────┘
              │
        ┌─────▼──────────┐
        │ SUCCESS        │
        │ Model Found    │
        │ At Any Level   │
        └────────────────┘
```

---

## Platform Compatibility Before & After

### BEFORE (Current State)

```
WINDOWS          LINUX            MAC              DOCKER
❌ CRASH        ✅ WORKS         ❌ CRASH         ⚠️ MAYBE
Path /home      Path exists      Path /home       Depends
doesn't exist                     doesn't exist    on build
```

### AFTER (With Fixes)

```
WINDOWS          LINUX            MAC              DOCKER
✅ WORKS        ✅ WORKS         ✅ WORKS         ✅ WORKS
Env var or      Original path    Env var or       ENV in
relative path   (unchanged)      relative path    Dockerfile
```

---

## File Impact Map

### 🔴 CRITICAL FILES (Break Windows)

```
ESI_Engine.py (MAIN ENGINE)
├─ Line 324: "/home/wtc/Videos/nursehelpdesk/..."
├─ Line 326: "/home/wtc/Videos/nursehelpdesk/...-003.gguf"
│
└─ IMPACT: System won't load on Windows at all
   └─ Application crashes on startup
      └─ Users can't run triage engine
```

### 🟡 HIGH PRIORITY FILES (Download Scripts)

```
20+ download_*.py files
├─ download_medgemma_model.py (line 24)
├─ download_llama_model.py (line 24)
├─ download_med_gemma_model.py (line 13)
├─ download_meditron_model.py (line 10)
├─ download_mistral_model.py (line 10)
└─ ... 15 more ...

IMPACT: Users can't download models on Windows
└─ ML inference disabled
   └─ Triage system falls back to deterministic rules only
```

### 🟠 MEDIUM PRIORITY FILES (Utilities)

```
inspect_pmc_parquet.py (lines 94-95)
├─ Hardcoded data paths
└─ IMPACT: Data inspection tools fail

inspect_faiss_data.py (line 227)
├─ Print statement with hardcoded path
└─ IMPACT: Debug output incorrect, confusing users

deepseek_integration_guide.py (lines 11, 27)
├─ Documentation with hardcoded paths
└─ IMPACT: Users can't follow setup guide on Windows
```

---

## Error Chain: What Happens on Windows

```
┌─────────────────────────────────────┐
│ User: python ESI_Engine.py          │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ Python loads ESI_Engine.py          │
│ Parses all imports and constants    │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ ESI_Engine.py line 324:             │
│ GGUF_MODEL_PATH = _GGUF_CANDIDATES │
│                                     │
│ Tries: "/home/wtc/Videos/nurse...  │
│ Result: ❌ Path doesn't exist!      │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ Later in code, tries to load model: │
│ llama_cpp.Llama(GGUF_MODEL_PATH)    │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 💥 CRASH: FileNotFoundError         │
│ "/home/wtc/Videos/.../model.gguf"  │
│ The system cannot find the path... │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ User gets: ERROR 500                │
│ (or doesn't get past startup)       │
│                                     │
│ Triage service: DOWN                │
└─────────────────────────────────────┘
```

---

## Solution Flow: Universal Path Detection

```
┌──────────────────────────────────────────────────────────┐
│ ESI_Engine.py starts                                     │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
      ┌────────────────────────────────┐
      │ Call: _get_gguf_model_path()   │
      │ (NEW UNIVERSAL FUNCTION)       │
      └────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      ┌──────┐    ┌──────┐    ┌──────┐
      │Env   │    │Relative   │Platform
      │Var   │    │Paths  │    │Specific
      │Check │    │Check  │    │Check
      └──┬───┘    └───┬───┘    └───┬──┘
         │            │            │
         │ EXISTS?    │ EXISTS?    │ EXISTS?
         │            │            │
         ├─YES─┐      │            │
         │     └──YES─┤            │
         │            └──YES────┐  │
         │                      │  │
         │                      ▼  ▼
         │                  ┌──────────┐
         │                  │ FOUND    │
         │                  │ GGUF     │
         │                  │ MODEL    │
         │                  └────┬─────┘
         │                       │
         └───────────────────────┤
                                 │
                                 ▼
                    ┌────────────────────┐
                    │ Return model path  │
                    │ ✅ Success         │
                    └────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────┐
                    │ Load GGUF model    │
                    │ Server starts      │
                    │ Ready for requests │
                    └────────────────────┘
```

---

## Environment Variable Configuration

### Windows Setup

```powershell
# Option 1: Set for current session
$env:GGUF_MODEL_PATH = "C:\Users\mbule\Videos\..."

# Option 2: Set permanently (System)
[Environment]::SetEnvironmentVariable(
    "GGUF_MODEL_PATH",
    "C:\Users\mbule\Videos\...",
    "User"
)

# Option 3: Create .env file
# Add to project root:
# GGUF_MODEL_PATH=C:\Users\mbule\Videos\...
```

### Linux/Mac Setup

```bash
# Option 1: Set for current session
export GGUF_MODEL_PATH="/home/wtc/Videos/..."

# Option 2: Set permanently
echo 'export GGUF_MODEL_PATH="/home/wtc/Videos/..."' >> ~/.bashrc
source ~/.bashrc

# Option 3: Create .env file
# Add to project root:
# GGUF_MODEL_PATH=/home/wtc/Videos/...
```

---

## Severity Level Visualization

```
┌─────────────────────────────────────────────────────────┐
│                    ISSUE SEVERITY                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🔴 CRITICAL (Causes System Crash)                      │
│    ├─ Hardcoded Linux paths in ESI_Engine.py           │
│    └─ Impact: Windows = Complete failure               │
│                                                         │
│ 🟡 HIGH (Breaks Major Features)                        │
│    ├─ Hardcoded paths in 20 download scripts           │
│    └─ Impact: Model downloads fail on Windows          │
│                                                         │
│ 🟠 MEDIUM (Affects User Experience)                    │
│    ├─ Colorama/Flask conflicts                         │
│    ├─ Path inconsistency                               │
│    └─ Impact: Console spam, debugging confusion        │
│                                                         │
│ 🟢 LOW (Edge Cases)                                    │
│    ├─ Missing psutil fallback                          │
│    ├─ Documentation paths                              │
│    └─ Impact: Unhelpful errors in edge cases           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Files You Should Read

```
START HERE ──→ SYSTEM_CONFLICT_AUDIT.md (this file)
              Executive summary + quick overview

FOR QUICK FIX ──→ QUICK_FIX_CROSS_PLATFORM.md
                  3 steps + 15 minutes to Windows support

FOR FULL DETAILS ──→ CROSS_PLATFORM_COMPATIBILITY_FIXES.md
                     Complete technical guide
                     Implementation checklist
                     Code examples for all fixes

FOR AUTOMATION ──→ fix_cross_platform.py
                   Run: python fix_cross_platform.py
                   Automatically fixes all 20+ files
```

---

## Quick Status Check

Run this to see what's broken:

```bash
# Windows PowerShell
echo "Checking path existence:"
Test-Path "C:\Users\mbule\Videos\nursehelpdesk\backend\ML\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\medgemma-4b-it-Q4_K_M.gguf"

# Linux/Mac
echo "Checking path existence:"
test -f "/home/wtc/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M.gguf" && echo "EXISTS" || echo "MISSING"
```

Expected: `True` or `EXISTS` = Model file is where code expects it

---

## Next Steps

1. ✅ **Read:** This summary (you're doing it!)
2. ⏭️  **Choose:** Quick fix (15 min) OR Full fix (1 hour)
3. ⏭️  **Implement:** Follow QUICK_FIX_CROSS_PLATFORM.md
4. ⏭️  **Test:** Run ESI_Engine.py on Windows
5. ⏭️  **Verify:** Check logs for `✅ GGUF from ...`

**Time to Windows compatibility: 15 minutes**
