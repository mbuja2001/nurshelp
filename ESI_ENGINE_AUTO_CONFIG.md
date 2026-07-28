# ESI_Engine Auto-Configuration

## Overview

The configuration detection logic is now **embedded directly in ESI_Engine.py**. When you run the engine, it automatically detects your system specifications and applies optimal settings without requiring a separate setup step.

## How It Works

### Startup Sequence

When you run `python ESI_Engine.py`:

1. **Auto-Configuration Engine Activates** (lines 30-130 in ESI_Engine.py)
   - Detects system CPU cores, RAM, GPU availability
   - Scans for GGUF models in `./models/` directory
   - Scans for RAG systems in `./knowledge_graph/` directory
   - Applies optimal performance parameters

2. **Configuration Priority**
   - If `triage_config.json` exists in the backend/ML directory → Use saved config
   - Otherwise → Auto-detect and apply settings

3. **Environment Variables Auto-Set**
   - `GGUF_N_THREADS`: Optimal threading (4-12 based on CPU cores)
   - `GGUF_N_BATCH`: Batch size (1024-2000 based on model size)
   - `GGUF_N_CTX`: Context window (1200-4096 based on model size)
   - `GGUF_USE_MLOCK`: Memory locking if RAM > 8GB
   - `DISABLE_RAG`: Auto-set if CPU-only with <16GB RAM

4. **Model Detection**
   - Finds largest GGUF model by file size
   - Estimates model type (4B/7B/13B) from file size
   - Calculates performance parameters accordingly

5. **Hardware Optimization**
   - **GPU Detection**: Automatically enables hybrid (RAG+LLM) mode if CUDA available
   - **CPU Optimization**: Scales thread count from 4 to 12+ based on available cores
   - **Memory Safety**: Disables RAG on CPU-only systems with insufficient RAM

6. **Console Output**
   - Prints `[CONFIG]` messages showing detected hardware and applied settings
   - Prints enhanced `[startup]` summary with all configuration details

## Startup Output Example

```
[CONFIG] 🔍 Auto-detecting system configuration...
[CONFIG] CPU: 8 logical cores (4 physical)
[CONFIG] RAM: 32.0GB total, 28.5GB available
[CONFIG] ✅ Found 1 GGUF model(s)
[CONFIG]    Largest: medgemma-7b-q4-k-m.gguf (5.2GB)
[CONFIG]    Estimated: 7B model (~18 tokens/sec on CPU)
[CONFIG] ✅ GGUF Configuration:
[CONFIG]    N_THREADS: 8
[CONFIG]    BATCH_SIZE: 1600
[CONFIG]    CTX_SIZE: 2048
[CONFIG]    USE_MLOCK: Yes
[CONFIG] ✅ RAG System detected
[CONFIG]    Index files: 2
[CONFIG]    Case databases: 5
[CONFIG]    RAG enabled (GPU or sufficient RAM)
[CONFIG] ✅ Auto-configuration complete

================================================================================
[startup] ✅ ESI_Engine Auto-Configuration & Initialization Complete
[startup]
[startup] 📋 Configuration Status:
[startup]    • Mode: HYBRID (RAG + LLM)
[startup]    • GPU Available: True
[startup]    • GGUF Model: ✅ Loaded
[startup]    • GGUF N_THREADS: 8
[startup]    • GGUF Batch Size: 1600
[startup]    • GGUF Context Size: 2048
[startup]    • RAG Status: ✅ Enabled
[startup]
[startup] 🔌 Available Endpoints:
[startup]    • POST /api/triage (hybrid, if GPU available)
[startup]    • POST /api/llm-triage (LLM-only, CPU mode)
[startup]    • GET /api/capabilities (inspect hardware & auto-config)
[startup]    • GET /health (health check)
[startup]
[startup] 🚀 Server Details:
[startup]    • Listening on 0.0.0.0:5000
[startup]    • Request deduplication: ENABLED (30s TTL)
[startup]    • Thread-safe GGUF: ENABLED (semaphore + lock)
[startup]    • Auto-configuration: ACTIVE (detects hardware on each run)
[startup]    • Press CTRL+C to shutdown gracefully
================================================================================
```

## Directory Structure Expected

```
backend/ML/
├── ESI_Engine.py                 # Main engine (with embedded auto-config)
├── models/
│   ├── medgemma-7b-q4-k-m.gguf  # GGUF models detected here
│   └── *.gguf
├── knowledge_graph/
│   ├── *.index                   # FAISS vector indexes
│   ├── *.jsonl                   # Case databases
│   └── *.json
└── triage_config.json            # Optional: saved config from setup wizard
```

## Model Size Mapping

| File Size | Model Type | Batch Size | Context | Est. Speed |
|-----------|-----------|-----------|---------|-----------|
| < 3.5GB  | 4B        | 2000      | 1200    | ~25 tok/s |
| 3.5-6GB  | 7B        | 1600      | 2048    | ~18 tok/s |
| > 6GB    | 13B+      | 1024      | 4096    | ~10 tok/s |

## Performance Modes

### Hybrid Mode (GPU + RAG)
- **Triggered**: GPU detected AND sufficient RAM
- **Endpoints**: `/api/triage`, `/api/llm-triage`
- **Benefits**: Fastest clinical reasoning with grounded case retrieval
- **Best For**: Hospital deployments with NVIDIA GPUs

### LLM-Only Mode (CPU)
- **Triggered**: No GPU OR CPU-only with <16GB RAM
- **Endpoints**: `/api/llm-triage`
- **Benefits**: Faster single inference on CPU systems
- **Best For**: Smaller deployments, development machines

## Overriding Auto-Configuration

### Option 1: Use Saved Config
Create `triage_config.json` in backend/ML directory:
```json
{
  "gguf_config": {
    "n_threads": 12,
    "n_batch": 2048,
    "n_ctx": 4096,
    "use_mlock": 1
  },
  "rag_config": {
    "enabled": true,
    "batch_retrieval_size": 512
  }
}
```

### Option 2: Environment Variables
```bash
# Disable RAG entirely
export DISABLE_RAG=1

# Or set specific GGUF parameters
export GGUF_N_THREADS=16
export GGUF_N_BATCH=4096
export GGUF_N_CTX=8192

# Then run the engine
python ESI_Engine.py
```

### Option 3: Direct Modification
Edit the `_auto_configure_engine()` function in ESI_Engine.py for custom logic.

## API Endpoints

### `/api/capabilities` (GET)
Returns detected hardware and configuration:
```json
{
  "gpu_available": true,
  "gguf_available": true,
  "rag_available": true,
  "gguf_n_threads": 8,
  "gguf_batch_size": 1600,
  "gguf_ctx_size": 2048,
  "triage_mode": "hybrid",
  "recommended_endpoint": "/api/triage"
}
```

### `/health` (GET)
Quick health check:
```json
{
  "ok": true,
  "gpu_available": true,
  "triage_mode": "hybrid",
  "recommended_endpoint": "/api/triage"
}
```

## Auto-Configuration Dependencies

- **psutil** (required): System resource detection
- **torch** (optional): GPU detection
- **cpuinfo** (optional): CPU model name

If psutil is missing, auto-configuration is skipped with a warning. The engine will still run but won't auto-optimize parameters.

## Troubleshooting

### Issue: "psutil not available - skipping auto-configuration"
**Solution**: Install psutil
```bash
pip install psutil
```

### Issue: Auto-config detects GPU but RAG not enabled
**Check**:
1. Is knowledge_graph directory present?
2. Do you have FAISS index files (*.index)?
3. Are there case databases (*.json or *.jsonl)?

**For CPU-only with <16GB RAM**: RAG is intentionally disabled for performance.

### Issue: N_THREADS set too high, causing slowdown
**Solution**: Create `triage_config.json` with custom threading or use environment variable:
```bash
export GGUF_N_THREADS=4
python ESI_Engine.py
```

## Logging

All auto-configuration messages are logged with `[CONFIG]` prefix:
- ✅ = Success (auto-detected and applied)
- ⚠️ = Warning (skipped or degraded mode)
- 🔍 = In progress
- ❌ = Error (check requirements)

## Summary

✨ **Zero-Configuration Startup**: Just run `python ESI_Engine.py` and the engine automatically optimizes itself for your hardware!
