# ESI_Engine Auto-Configuration Architecture

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     python ESI_Engine.py                        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ Auto-Config Engine   │  ← NEW: Runs at startup
        │ (_auto_configure_    │
        │  engine function)    │
        └────┬─────────────────┘
             │
      ┌──────┴──────┬──────────┬────────────┐
      ▼             ▼          ▼            ▼
   ┌──────┐    ┌──────────┐ ┌──────────┐ ┌──────┐
   │ Load │    │ Detect   │ │ Detect   │ │Check │
   │Saved │    │Hardware  │ │GGUF      │ │RAG   │
   │Config│    │(CPU/RAM) │ │Models    │ │System│
   └──────┘    └──────────┘ └──────────┘ └──────┘
      │             │          │            │
      └─────────────┴──────────┴────────────┘
              │
              ▼
       ┌──────────────────┐
       │ Calculate Optimal│
       │ Performance      │
       │ Parameters       │
       └────────┬─────────┘
                │
      ┌─────────┴─────────┐
      ▼                   ▼
  ┌─────────────┐   ┌──────────────────┐
  │ Set GGUF    │   │ Set RAG Mode     │
  │Environment  │   │ (DISABLE_RAG=0/1)│
  │Variables    │   └──────────────────┘
  └─────────────┘
      │
      ▼
  ┌──────────────────────┐
  │ Flask App Starts     │
  │ (Uses env variables) │
  └──────────────────────┘
      │
      ▼
  ┌──────────────────┐
  │ Print Detailed   │
  │ Startup Summary  │
  │ ([startup] msgs) │
  └──────────────────┘
      │
      ▼
  ┌──────────────────────────────────────┐
  │ Server Ready on 0.0.0.0:5000        │
  │ /api/triage (hybrid)                │
  │ /api/llm-triage (cpu)               │
  │ /api/capabilities (inspect config)  │
  └──────────────────────────────────────┘
```

## Startup Sequence (Detailed)

### Phase 1: Module Initialization (Lines 1-160)
```python
# ESI_Engine.py imports
from flask import Flask, request, jsonify
import os
from pathlib import Path
# ... other imports ...

# Offline caching setup
_base_dir = Path(__file__).resolve().parent
_cache_dir = _base_dir / "model_cache"
os.environ["HF_HUB_OFFLINE"] = "1"  # Prevent internet requests

# ← NEW: Auto-configuration function added here (lines 30-130)
def _auto_configure_engine():
    # Detects hardware, GGUF models, RAG systems
    # Sets environment variables
    pass

# ← NEW: Auto-config runs on import (line ~137)
_auto_configure_engine()

# Now LLM_ONLY_MODE uses auto-configured DISABLE_RAG variable
LLM_ONLY_MODE = bool(os.environ.get("DISABLE_RAG", "0") in ("1", "true", "True"))
```

### Phase 2: Model Loading
```python
# GPU detection (lines 160-180)
_GPU_AVAILABLE = torch.cuda.is_available()

# RAG initialization (uses env vars from auto-config)
if not LLM_ONLY_MODE:
    RAG_RETRIEVER = TriageRAGRetriever()
    
# GGUF model loading (uses GGUF_N_THREADS, GGUF_N_BATCH, etc.)
_GGUF_AVAILABLE = True
```

### Phase 3: API Routes Registered
```python
@app.route('/api/triage', methods=['POST'])
@app.route('/api/llm-triage', methods=['POST'])
@app.route('/api/capabilities', methods=['GET'])
@app.route('/health', methods=['GET'])
```

### Phase 4: Startup Handler
```python
if __name__ == '__main__':
    # Print enhanced startup summary (uses env vars)
    print(f"[startup] GGUF N_THREADS: {os.environ.get('GGUF_N_THREADS', 'auto')}")
    print(f"[startup] RAG Status: {'✅ Enabled' if RAG_AVAILABLE else '❌ Disabled'}")
    
    # Start Werkzeug server with threading
    server = make_server('0.0.0.0', BACKEND_PORT, app, threaded=True)
    server.serve_forever()
```

## Auto-Configuration Decision Tree

```
START (ESI_Engine.py runs)
│
├─ Check for triage_config.json
│  ├─ YES → Load and apply config, RETURN
│  └─ NO → Continue
│
├─ Detect system (psutil)
│  ├─ Get CPU cores, RAM, GPU
│  └─ Continue
│
├─ Detect GGUF models (./models/*.gguf)
│  ├─ Found?
│  │  ├─ YES → Estimate model size → Set BATCH, CTX, THREADS
│  │  └─ NO → Use defaults
│  └─ Continue
│
├─ Detect RAG system (./knowledge_graph/*)
│  ├─ Found?
│  │  ├─ YES → Check GPU availability
│  │  │  ├─ GPU? → RAG enabled
│  │  │  └─ CPU-only + <16GB RAM? → DISABLE_RAG=1
│  │  └─ NO → Continue
│  └─ Continue
│
├─ Set all environment variables
│  └─ GGUF_N_THREADS, GGUF_N_BATCH, GGUF_N_CTX, GGUF_USE_MLOCK, DISABLE_RAG
│
└─ RETURN & Continue with Flask initialization
```

## Configuration Priority (Cascade)

### Level 1: Saved Configuration (Highest Priority)
```json
// triage_config.json in backend/ML directory
{
  "gguf_config": { "n_threads": 16, ... },
  "rag_config": { "enabled": false }
}
```

### Level 2: Environment Variables
```bash
GGUF_N_THREADS=12
DISABLE_RAG=1
```

### Level 3: Auto-Detected (Lowest Priority)
```python
# _auto_configure_engine() calculates these
GGUF_N_THREADS = max(4, min(cpu_cores_logical, 12))
GGUF_N_BATCH = 1600 if 3.5GB < model < 6GB else ...
```

## Key Detection Algorithms

### 1. Model Type Estimation
```python
model_size_gb = file_size / (1024^3)

if model_size_gb < 3.5:
    model_type = "4B"
    batch_size = 2000
    ctx_size = 1200
    estimated_tps = 25  # tokens/sec
elif model_size_gb < 6:
    model_type = "7B"
    batch_size = 1600
    ctx_size = 2048
    estimated_tps = 18
else:
    model_type = "13B+"
    batch_size = 1024
    ctx_size = 4096
    estimated_tps = 10
```

### 2. Optimal Thread Count
```python
cpu_count_logical = psutil.cpu_count(logical=True)
n_threads = max(4, min(cpu_count_logical, 12))

# Logic:
# - Low-end (≤2 cores): 4 threads
# - Mid-range (4-8 cores): match logical count
# - High-end (>12 cores): cap at 12 (diminishing returns)
```

### 3. Memory Safety
```python
ram_available_gb = psutil.virtual_memory().available / (1024^3)

use_mlock = ram_available_gb > 8  # Memory locking if plenty of RAM
disable_rag = (not gpu_available) and (ram_available_gb < 16)  # CPU-only safety
```

## Data Structures

### Auto-Config Output
```python
{
    "gguf_detected": bool,
    "gguf_files": [
        {
            "name": "medgemma-7b-q4-k-m.gguf",
            "size_gb": 5.2,
            "estimated_type": "7B"
        }
    ],
    "system_specs": {
        "cpu_cores": 8,
        "ram_gb": 32.0,
        "gpu_available": True
    },
    "rag_detected": bool,
    "optimization": {
        "n_threads": 8,
        "n_batch": 1600,
        "n_ctx": 2048,
        "use_mlock": True
    }
}
```

## Environment Variables Generated

| Variable | Source | Value | Used By |
|----------|--------|-------|---------|
| `GGUF_N_THREADS` | Auto-config | 4-12 | GGUF inference |
| `GGUF_N_BATCH` | Auto-config | 1024-2000 | GGUF batching |
| `GGUF_N_CTX` | Auto-config | 1200-4096 | Context window |
| `GGUF_USE_MLOCK` | Auto-config | 0/1 | Memory pinning |
| `DISABLE_RAG` | Auto-config | 0/1 | RAG mode toggle |

## Integration Points

### 1. RAG Retriever
```python
if LLM_ONLY_MODE:  # Set by DISABLE_RAG
    RAG_RETRIEVER = None  # Skip RAG
else:
    RAG_RETRIEVER = TriageRAGRetriever()  # Use RAG
```

### 2. GGUF Loading
```python
# Uses GGUF_N_THREADS, GGUF_N_BATCH, GGUF_N_CTX
model = Llama(
    model_path="./models/medgemma-7b.gguf",
    n_threads=int(os.environ["GGUF_N_THREADS"]),
    n_batch=int(os.environ["GGUF_N_BATCH"]),
    n_ctx=int(os.environ["GGUF_N_CTX"])
)
```

### 3. API Capabilities
```python
@app.route('/api/capabilities')
def capabilities():
    return {
        "gguf_n_threads": os.environ.get("GGUF_N_THREADS"),
        "gguf_batch_size": os.environ.get("GGUF_N_BATCH"),
        "gguf_ctx_size": os.environ.get("GGUF_N_CTX"),
        "rag_enabled": not LLM_ONLY_MODE,
        "gpu_available": _GPU_AVAILABLE
    }
```

## Performance Implications

### CPU Threading
- **4 threads**: Single-core machines (light load)
- **8 threads**: Standard quad-core (balanced)
- **12+ threads**: High-end systems (parallelizable workloads)

### Batch Size
- **Larger batches** (2000): 4B models on high-RAM systems
- **Medium batches** (1600): 7B models (balanced VRAM/speed)
- **Smaller batches** (1024): 13B models (memory safety)

### Context Window
- **1200 tokens**: 4B models (fast, limited context)
- **2048 tokens**: 7B models (standard medical notes)
- **4096 tokens**: 13B models (full case summaries)

### Memory Locking (mlock)
- **Enabled** (>8GB available): Prevent page swapping, faster inference
- **Disabled** (<8GB): Save RAM for other processes

## Fallback Behavior

If auto-configuration fails at any step:
1. **psutil missing?** → Skip auto-config, use hardcoded defaults
2. **No GGUF models?** → Proceed (may fail later if GGUF required)
3. **No RAG system?** → Proceed with LLM-only mode
4. **GPU detection fails?** → Assume CPU-only
5. **Hardware detection fails?** → Use conservative defaults

## Testing Auto-Configuration

### Verify Environment Variables Set
```bash
# After engine starts, inspect /api/capabilities endpoint
curl http://localhost:5000/api/capabilities
```

### Check Console Output
```
[CONFIG] 🔍 Auto-detecting system configuration...
[CONFIG] CPU: 8 logical cores (4 physical)
[CONFIG] RAM: 32.0GB total, 28.5GB available
[CONFIG] ✅ Found 1 GGUF model(s)
[CONFIG] ✅ GGUF Configuration:
[CONFIG]    N_THREADS: 8
[CONFIG]    BATCH_SIZE: 1600
[CONFIG] ✅ Auto-configuration complete
```

---

**Summary**: Auto-configuration eliminates manual setup while maintaining override flexibility via environment variables or saved configs. The engine adapts to any hardware configuration automatically.
