# Quick Start: ESI_Engine with Auto-Configuration

## One-Command Startup

```bash
cd c:\Users\mbule\Videos\nursehelpdesk\Nurshelp\backend\ML
python ESI_Engine.py
```

✨ **That's it!** The engine will:
1. Detect your system (CPU cores, RAM, GPU)
2. Find GGUF models in `./models/`
3. Find RAG knowledge graph in `./knowledge_graph/`
4. Apply optimal performance settings
5. Start the API server

## What Happens on Startup

### Console Output
```
[CONFIG] 🔍 Auto-detecting system configuration...
[CONFIG] CPU: 8 logical cores (4 physical)
[CONFIG] RAM: 32.0GB total, 28.5GB available
[CONFIG] ✅ Found 1 GGUF model(s)
[CONFIG]    Largest: medgemma-7b-q4-k-m.gguf (5.2GB)
[CONFIG] ✅ GGUF Configuration:
[CONFIG]    N_THREADS: 8
[CONFIG]    BATCH_SIZE: 1600
[CONFIG] ✅ Auto-configuration complete

================================================================================
[startup] ✅ ESI_Engine Auto-Configuration & Initialization Complete
[startup] 📋 Configuration Status:
[startup]    • Mode: HYBRID (RAG + LLM)
[startup]    • GPU Available: True
[startup]    • GGUF Model: ✅ Loaded
[startup] 🔌 Available Endpoints:
[startup]    • POST /api/triage (hybrid mode)
[startup]    • POST /api/llm-triage (CPU-only)
[startup]    • GET /api/capabilities (inspect config)
[startup] 🚀 Listening on 0.0.0.0:5000
================================================================================
```

## Required Directory Structure

```
backend/ML/
├── ESI_Engine.py
├── models/
│   └── medgemma-7b-q4-k-m.gguf    ← GGUF models detected here
├── knowledge_graph/
│   ├── graph_nodes*.json            ← Optional: knowledge graph
│   ├── graph_edges*.json
│   └── *.index                      ← Optional: FAISS indexes
```

## Auto-Configuration Features

✅ **CPU Detection** → Optimal threading (4-12 cores)
✅ **RAM Detection** → Batch size optimization & memory safety
✅ **GPU Detection** → Automatic hybrid mode if CUDA available
✅ **GGUF Detection** → Model size → Performance mapping
✅ **RAG Detection** → Enables/disables based on hardware

## Override Auto-Configuration

### Option 1: Environment Variables
```bash
set DISABLE_RAG=1
set GGUF_N_THREADS=16
python ESI_Engine.py
```

### Option 2: Saved Config (triage_config.json)
Place in backend/ML directory:
```json
{
  "gguf_config": {
    "n_threads": 12,
    "n_batch": 2048,
    "n_ctx": 4096
  },
  "rag_config": {
    "enabled": false
  }
}
```

## Test the API

### Check Configuration
```bash
curl http://localhost:5000/api/capabilities
```

### Check Health
```bash
curl http://localhost:5000/health
```

### Run Triage
```bash
curl -X POST http://localhost:5000/api/triage \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P12345",
    "age_years": 45,
    "vitals": {"temp": 38.5, "bp": "140/90", "hr": 110},
    "chief_complaint": "fever and chest pain"
  }'
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `psutil not available` | `pip install psutil` |
| Port 5000 already in use | `set BACKEND_PORT=5001` or kill existing process |
| No GGUF models found | Create `models/` directory and add `*.gguf` files |
| RAG disabled unexpectedly | Check if CPU-only with <16GB RAM (auto-safety feature) |

## Files Modified

- **ESI_Engine.py**: Added `_auto_configure_engine()` function + enhanced startup messages
- **No breaking changes**: All existing functionality preserved

## Key Environment Variables Set by Auto-Config

```
GGUF_N_THREADS          → CPU thread count (auto-detected)
GGUF_N_BATCH            → Batch size for inference
GGUF_N_CTX              → Context window size
GGUF_USE_MLOCK          → Memory locking for speed
DISABLE_RAG             → Set to "1" if RAG disabled
```

Query them with: `curl http://localhost:5000/api/capabilities`

---

**Zero-configuration deployment!** Just run the engine and it adapts to your hardware. 🚀
