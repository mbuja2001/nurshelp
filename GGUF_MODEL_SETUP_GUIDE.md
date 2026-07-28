# 📁 GGUF Model Directory Setup Guide

## Quick Status

✅ **Server is running successfully** on http://0.0.0.0:5000  
✅ **OS-Aware Configuration Working:** Windows detected  
✅ **Hardware Tier Selected:** TIER 0 (Ultra Low, for your 0.7GB available RAM)  
⚠️ **GGUF Model:** Not loaded yet (system running in LLM-only fallback mode)

---

## What You Need to Do: Place the GGUF Model

### The Problem
Your system is ready but needs the actual GGUF model file to load. Currently showing:
```
[GGUF] ⚠️  GGUF file not found
[startup] ⚠️  GGUF not available: falling back to deterministic rules only
```

### The Solution
Download the model and place it in the **correct directory structure**.

---

## Directory Structure: What to Create

### Option A: Using Relative Paths (Recommended - Windows)
```
C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\
├── core\
│   └── 2_engine.py                          ← Engine file
│
└── data\                                    ← Create this
    └── models\                              ← Create this
        └── medgemma-4b-local\               ← Create this
            └── models--unsloth--medgemma-4b-it-bnb-4bit\  ← Create this
                └── medgemma-4b-it-Q4_K_M-003.gguf       ← Place model here
```

### Option B: Using Different Location
```
C:\Models\                                  ← Could be anywhere
└── medgemma-4b-local\                      ← Required subdirectory
    └── models--unsloth--medgemma-4b-it-bnb-4bit\  ← Required subdirectory
        └── medgemma-4b-it-Q4_K_M-003.gguf       ← Place model here
```

---

## Step-by-Step Setup

### Step 1: Create the Directory Structure

**Windows (PowerShell or Command Prompt):**
```bash
cd C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete

# Create directories
mkdir data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit
```

**Windows (File Explorer):**
1. Navigate to: `C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\`
2. Create folder: `data`
3. Inside `data`, create: `models`
4. Inside `models`, create: `medgemma-4b-local`
5. Inside `medgemma-4b-local`, create: `models--unsloth--medgemma-4b-it-bnb-4bit`

### Step 2: Download the GGUF Model

**Option A: Using HuggingFace CLI**
```bash
cd C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete

# Download to the correct location
huggingface-cli download unsloth/medgemma-4b-it-GGUF \
  medgemma-4b-it-Q4_K_M.gguf \
  --local-dir data/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit
```

**Option B: Manual Download**
1. Go to: https://huggingface.co/unsloth/medgemma-4b-it-GGUF
2. Download file: `medgemma-4b-it-Q4_K_M.gguf` (~2.4GB)
3. Place in: `data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\`

**Option C: From Existing Location**
If you already have the model elsewhere:
```bash
copy "C:\Downloads\medgemma-4b-it-Q4_K_M.gguf" ^
     "C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\"
```

### Step 3: Verify the Structure

**Check with PowerShell:**
```bash
# Should show the .gguf file
dir data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\*.gguf
```

**Output should show:**
```
    Directory: C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---           6/5/2026 10:00 AM   2500000000 medgemma-4b-it-Q4_K_M-003.gguf
```

### Step 4: Restart the Engine

**Stop current server:**
```bash
# Press Ctrl+C in the terminal
```

**Start again:**
```bash
cd ESI_Engine_Complete
python -m core.2_engine
```

**Should now show:**
```
[GGUF] Loading GGUF model with hardware-optimized parameters...
[GGUF] Model: C:\Users\...\data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\medgemma-4b-it-Q4_K_M-003.gguf
[GGUF] ✅ Model loaded successfully
[startup] ✅ HYBRID PIPELINE ENABLED
[startup] Starting on 0.0.0.0:5000
```

---

## What Each Directory Means

```
data/                          ← Engine data (relative to script)
└── models/                    ← GGUF model storage
    └── medgemma-4b-local/     ← Model name (required)
        └── models--unsloth--medgemma-4b-it-bnb-4bit/  ← Model variant (required)
            └── medgemma-4b-it-Q4_K_M-003.gguf        ← Actual model file
```

**Why this structure?**
- `data/` keeps everything organized relative to the engine
- `models/` separates from other data (RAG, knowledge graphs, etc.)
- `medgemma-4b-local/` is the model version/variant name
- `models--unsloth--medgemma-4b-it-bnb-4bit/` is the HuggingFace model ID format
- The `.gguf` file is the actual quantized model

---

## Alternative: Provide Path When Prompted

If you place the model elsewhere, you can use the interactive prompt:

**When you see:**
```
[PROMPT] Where is your GGUF model directory?
   Windows examples:
     • Relative: .\data\models
     • Absolute: C:\path\to\models

   Enter path (or press Enter to skip):
```

**You can provide:**
```
# Option 1: The models parent directory
C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\data\models

# Option 2: The medgemma-4b-local directory
C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\data\models\medgemma-4b-local

# Option 3: The models--unsloth-- directory
C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit

# Option 4: Any parent directory (system will search recursively)
C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\
```

**The system will now:**
1. Check if you provided the `models/` directory
2. Check if you provided the `medgemma-4b-local/` directory  
3. Check if you provided the `models--unsloth--*/` directory
4. Recursively search for the `.gguf` file in subdirectories
5. Avoid duplicate paths (the bug we just fixed!)

---

## File Size & Download Time

- **Model File:** medgemma-4b-it-Q4_K_M-003.gguf
- **Size:** ~2.4 GB
- **Download Time:**
  - 100 Mbps: ~3 minutes
  - 50 Mbps: ~6 minutes
  - 20 Mbps: ~16 minutes
  - Mobile: Longer (use wired if possible)

---

## After Model is Loaded

### What Changes

**Before (Current):**
```
[startup] ⚠️  GGUF not available: falling back to deterministic rules only
[startup] Mode: LLM-ONLY (CPU-mode)
[startup] GGUF Model: ❌ Not available
```

**After (When Model Loads):**
```
[startup] ✅ GGUF Model Ready
[startup] Mode: HYBRID (CPU-mode with LLM fallback)
[startup] GGUF Model: ✅ Available
```

### API Endpoints Available

**Current (LLM-only):**
- `POST /api/llm-triage` - LLM-only triage (works now)
- `GET /health` - Health check (works now)

**After Model Loads (Hybrid):**
- `POST /api/triage` - Hybrid triage (GGUF + LLM fallback)
- `POST /api/llm-triage` - LLM-only (still available)
- `GET /api/capabilities` - Hardware info
- `GET /health` - Health check

---

## Test the Setup

### Current (Without Model)
```bash
# This works NOW
curl -X POST http://localhost:5000/api/llm-triage \
  -H "Content-Type: application/json" \
  -d '{"patient_complaint": "chest pain"}'
```

### After Model Loads
```bash
# This will also work (better reasoning)
curl -X POST http://localhost:5000/api/triage \
  -H "Content-Type: application/json" \
  -d '{"patient_complaint": "chest pain"}'
```

---

## Troubleshooting

### Problem: "Path does not exist"
**Solution:** Make sure the full directory path exists:
```bash
# Verify each level exists
dir data
dir data\models
dir data\models\medgemma-4b-local
dir data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit
```

### Problem: "Model file not found"
**Solution:** Verify the .gguf file exists:
```bash
# Should list the .gguf file
dir data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit\*.gguf

# File should be ~2.4 GB
```

### Problem: System still shows "GGUF not available" after restart
**Solution:** 
1. Stop the server (Ctrl+C)
2. Verify model file exists
3. Check no typos in directory names
4. Restart: `python -m core.2_engine`

### Problem: Too slow or out of memory on weak hardware
**Note:** Your system detected TIER 0 (only 0.7GB available RAM). This is very constrained!
- Context is reduced to 512 tokens
- Batch size is minimal (64)
- Inference will be slow (55+ seconds expected)
- Consider freeing up RAM before running (close Chrome, etc.)

---

## Summary

1. ✅ **Server is running** - Port 5000 is ready
2. ⏳ **Waiting for model** - Place the .gguf file in the right directory
3. 📁 **Directory structure:**
   ```
   data/models/medgemma-4b-local/models--unsloth--medgemma-4b-it-bnb-4bit/medgemma-4b-it-Q4_K_M-003.gguf
   ```
4. 🔄 **Restart the engine** - `python -m core.2_engine`
5. ✅ **All set!** - API will now use GGUF for better reasoning

---

**Total Setup Time:** ~20-30 minutes (mostly download time)  
**Your Hardware:** Windows, 7.7GB total, 0.7GB available → TIER 0 (ultra-low)  
**Status:** Ready for model placement ✅
