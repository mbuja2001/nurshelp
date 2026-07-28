# 🏗️ Architecture & System Design

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Your Application Code                           │
│  (app.py, ESI_Engine.py, or any medical triage code)               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ from adaptable_engine import get_config
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AdaptableTriageEngine                             │
│  • Auto-detects platform                                            │
│  • Discovers hardware                                               │
│  • Finds models                                                     │
│  • Optimizes performance                                            │
│  • Exports configuration                                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ Uses
                             │
                ┌────────────┴──────────────┐
                │                           │
                ▼                           ▼
    ┌─────────────────────┐    ┌────────────────────┐
    │ cross_platform      │    │ External Libraries │
    │ _config.py          │    │ • psutil (hardware)│
    │ • OS Detection      │    │ • torch (GPU)      │
    │ • Path Handling     │    │ • colorama         │
    │ • Colorama Wrapper  │    │ • python-dotenv    │
    │ • Env Vars          │    │ • flask            │
    └─────────────────────┘    └────────────────────┘
                │
                │ Detects
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
    ┌─────────┐     ┌──────────┐
    │ Windows │     │ Linux    │
    └─────────┘     └──────────┘
        │                │
        │ File System    │
        │                │
    ┌───┴────┐      ┌────┴───┐
    │ Models │      │ Models │
    │ Logs   │      │ Logs   │
    │ Cache  │      │ Cache  │
    └────────┘      └────────┘
```

---

## Data Flow Diagram

```
STARTUP
─────────────────────────────────────────────────────────────

User imports adaptable_engine
        │
        ▼
Initialize cross-platform utilities
        │
        ▼
Create AdaptableTriageEngine instance
        │
        ├─► Detect Platform
        │   └─► Store OS, Python version, paths
        │
        ├─► Detect Hardware
        │   ├─► Query CPU info (psutil)
        │   ├─► Query RAM (psutil)
        │   └─► Query GPU (torch.cuda)
        │
        ├─► Configure Models
        │   ├─► Search for GGUF files
        │   ├─► Check RAG availability
        │   └─► Verify transformers cache
        │
        ├─► Calculate Performance
        │   ├─► CPU cores → threading config
        │   ├─► RAM → memory budgets
        │   ├─► GPU available → batch sizes
        │   └─► CPU cores → CPU batch sizes
        │
        └─► Print Summary
            └─► Display detected configuration

RUNTIME
─────────────────────────────────────────────────────────────

User calls get_config()
        │
        ▼
Configuration Dictionary
        │
        ├─► platform (Windows/Linux/Mac)
        ├─► hardware (CPU, GPU, RAM)
        ├─► models (GGUF, RAG, transformers)
        ├─► environment (device, offline, etc)
        ├─► features (what's available)
        └─► performance (tuning parameters)
        │
        ▼
Application uses configuration
        │
        ├─► Sets device (GPU/CPU)
        ├─► Sets batch size (GPU/CPU)
        ├─► Resolves model path
        ├─► Creates logs directory
        ├─► Uses thread count
        └─► Enables colorama for console
```

---

## Configuration Hierarchy

```
AdaptableTriageEngine.config
│
├─ platform {
│   ├─ system: "Windows" | "Linux" | "Darwin"
│   ├─ is_windows: bool
│   ├─ is_linux: bool
│   ├─ path_separator: "\\" | "/"
│   ├─ line_separator: "\r\n" | "\n"
│   └─ python_version: "3.11"
│
├─ hardware {
│   ├─ cpu_cores: 16
│   ├─ physical_cores: 8
│   ├─ total_ram_gb: 32.0
│   ├─ available_ram_gb: 24.5
│   ├─ gpu_available: true | false
│   ├─ gpu_type: "cuda" | "rocm" | "none"
│   ├─ gpu_vram_gb: 8.0
│   ├─ cuda_available: true | false
│   └─ cuda_version: "11.8"
│
├─ models {
│   ├─ gguf {
│   │   ├─ available: true | false
│   │   ├─ model_name: "medgemma-4b-it-Q4_K_M"
│   │   ├─ model_path: "/path/to/model.gguf"
│   │   ├─ model_size_gb: 2.5
│   │   ├─ quantization: "Q4_K_M"
│   │   ├─ context_size: 1200
│   │   └─ batch_size: 2000
│   ├─ rag {
│   │   ├─ available: true | false
│   │   ├─ retriever_type: "pubmedbert"
│   │   ├─ index_path: "/path/to/index"
│   │   └─ similarity_threshold: 0.70
│   └─ transformers {
│       ├─ available: true | false
│       └─ cache_location: "/path/to/cache"
│
├─ paths {
│   ├─ workspace_root: "/path/to/workspace"
│   ├─ backend_dir: "/path/to/backend"
│   ├─ ml_dir: "/path/to/backend/ML"
│   ├─ model_paths: [list of search paths]
│   ├─ cache_dir: "/path/to/cache"
│   ├─ logs_dir: "/path/to/logs"
│   └─ temp_dir: "/tmp" | "C:\\Temp"
│
├─ environment {
│   ├─ offline_mode: true
│   ├─ debug: false | true
│   ├─ backend_port: 5000
│   ├─ device: "cuda" | "cpu"
│   ├─ disable_rag: false | true
│   ├─ force_pure_llm: false | true
│   └─ environment: "development" | "production"
│
├─ features {
│   ├─ gguf_inference: true | false
│   ├─ rag_retrieval: true | false
│   ├─ gpu_acceleration: true | false
│   ├─ colorama_support: true
│   └─ offline_operation: true
│
└─ performance {
    ├─ max_threads: 8
    ├─ inference_threads: 6
    ├─ batch_size: 2000
    ├─ context_size: 1200
    ├─ use_memory_mapping: true
    ├─ memory_budget_pct: 90
    ├─ gpu_batch_size: 32
    ├─ gpu_memory_fraction: 0.8
    └─ use_mixed_precision: true
```

---

## Deployment Flow

```
USER RUNS DEPLOYMENT SCRIPT
│
├─ Windows: powershell -ExecutionPolicy Bypass -File deploy-windows.ps1
└─ Linux: bash deploy-linux.sh
    │
    ▼
DETECTION PHASE
    │
    ├─► Detect OS (Windows / Linux / macOS)
    ├─► Detect Python version
    ├─► Check for GPU (nvidia-smi)
    ├─► Count CPU cores
    └─► Get available RAM
    │
    ▼
ENVIRONMENT SETUP PHASE
    │
    ├─► Create virtual environment
    ├─► Upgrade pip
    ├─► Install requirements.txt
    ├─► Install GPU support (if GPU available)
    └─► Create directories (models, logs, cache)
    │
    ▼
VALIDATION PHASE
    │
    ├─► Verify cross-platform files exist
    ├─► Run adaptable_engine.py
    ├─► Check all dependencies installed
    └─► Validate configuration
    │
    ▼
COMPLETION
    │
    ├─► Create requirements.lock (for sync)
    ├─► Print summary
    └─► Ready for use!
```

---

## Class Hierarchy

```
CROSS_PLATFORM_CONFIG
│
├─ IS_WINDOWS (bool)
├─ IS_LINUX (bool)
├─ IS_MAC (bool)
├─ PLATFORM (dict)
│
├─ Colorama (class)
│   ├─ init()
│   ├─ colored(text, color)
│   ├─ print_status(msg, level)
│   └─ disable_if_needed()
│
├─ CrossPlatformPath (class)
│   ├─ join(*parts)
│   ├─ normalize(path)
│   └─ ensure_dir(path)
│
├─ EnvironmentConfig (class)
│   ├─ get_env_var(key, default)
│   └─ set_env_var(key, value)
│
├─ PythonEnvironment (class)
│   ├─ get_version()
│   ├─ get_executable()
│   └─ get_prefix()
│
└─ GGUFPathResolver (class)
    ├─ search_paths()
    └─ get_gguf_path()

ADAPTABLE_ENGINE
│
├─ get_engine() → AdaptableTriageEngine
├─ get_config() → dict
└─ print_config_summary() → None
    │
    └─ AdaptableTriageEngine (class)
        ├─ __init__()
        ├─ initialize() → bool
        ├─ get_config() → dict
        ├─ get_summary() → str
        ├─ _resolve_paths() → dict
        ├─ _detect_hardware() → dict
        ├─ _configure_models() → dict
        ├─ _configure_environment() → dict
        ├─ _configure_features() → dict
        └─ _configure_performance() → dict
```

---

## Integration Points

```
YOUR FLASK APP
│
├─► import get_config from adaptable_engine
│   └─► @app.route('/api/health')
│       └─► return get_config()['platform']
│
└─► Run: python -m flask run
    └─► Backend listens on config['environment']['backend_port']

YOUR ML ENGINE (ESI_Engine.py)
│
├─► import get_config from adaptable_engine
│   └─► config = get_config()
│
├─► Use device: config['environment']['device']
├─► Use batch: config['performance']['batch_size']
├─► Use model: config['models']['gguf']['model_path']
├─► Use threads: config['performance']['inference_threads']
└─► Use GPU settings if config['hardware']['gpu_available']

ANY PYTHON CODE
│
├─► from adaptable_engine import get_config
│   └─► config = get_config()
│
└─► Access any auto-detected setting for your purpose
```

---

## Error Handling Flow

```
Error Occurs During:
│
├─ Deployment Script
│   ├─ Python not found
│   │   └─ Show install instructions
│   ├─ GPU not detected
│   │   └─ Continue with CPU
│   ├─ Requirements install fails
│   │   └─ Show pip error + suggestions
│   └─ Virtual environment fails
│       └─ Show filesystem error + cleanup
│
├─ Configuration Detection
│   ├─ GGUF not found
│   │   └─ Mark available=false, continue
│   ├─ GPU not detected
│   │   └─ Use CPU settings
│   └─ Path issues
│       └─ Fall back to alternatives
│
└─ Runtime
    ├─ Model fails to load
    │   └─ Check model_path in config
    ├─ GPU memory insufficient
    │   └─ Reduce batch_size
    └─ Offline mode violated
        └─ Error and exit (medical safety)
```

---

## Configuration Decision Tree

```
START
│
├─ IS WINDOWS?
│   ├─ YES → Use PowerShell paths (C:\Users\...)
│   │        Use ANSI + Colorama for colors
│   │        Use UTF-8 line endings
│   └─ NO → Continue
│
├─ IS LINUX?
│   ├─ YES → Use Unix paths (/home/...)
│   │        Use native ANSI colors
│   │        Use LF line endings
│   └─ NO → Continue
│
├─ IS MACOS?
│   ├─ YES → Use Unix paths (/Users/...)
│   │        Use native ANSI colors
│   │        Use LF line endings
│   └─ NO → Error (unsupported)
│
├─ HAS GPU?
│   ├─ YES → Set device='cuda'
│   │        Use batch_size=32
│   │        Enable mixed precision
│   │        Use GPU memory fraction
│   └─ NO → Set device='cpu'
│           Use batch_size=2000
│           Use threading
│
├─ MEMORY CHECK
│   ├─ <8GB → Reduce batch sizes
│   ├─ 8-16GB → Standard batch sizes
│   └─ >16GB → Increase batch sizes
│
├─ CPU CORES CHECK
│   ├─ <4 cores → Reduce threads
│   ├─ 4-8 cores → Standard threading
│   └─ >8 cores → Increase threads
│
└─ EXPORT CONFIGURATION
    └─ All settings ready for use
```

---

## Performance Tuning Flowchart

```
PERFORMANCE CONFIGURATION
│
├─ If GPU Available (cuda):
│   ├─ gpu_batch_size = 32
│   ├─ gpu_memory_fraction = 0.8
│   ├─ use_mixed_precision = True
│   ├─ inference_threads = 2
│   └─ Use GPU for computation
│
├─ If CPU Only:
│   ├─ batch_size = 2000
│   ├─ inference_threads = physical_cores - 2
│   ├─ use_memory_mapping = True
│   └─ Use multi-threading
│
├─ Memory Adjustment:
│   ├─ available_ram < 8GB
│   │   └─ batch_size *= 0.5
│   ├─ available_ram 8-16GB
│   │   └─ batch_size *= 0.8
│   └─ available_ram > 16GB
│       └─ batch_size *= 1.2
│
└─ Final Performance Profile Ready
```

---

## File System Layout

```
WORKSPACE ROOT
│
├── nurshelp/
│   ├── backend/
│   │   ├── ✅ cross_platform_config.py (400 lines)
│   │   ├── ✅ adaptable_engine.py (500 lines)
│   │   ├── ✅ deploy-windows.ps1 (300 lines)
│   │   ├── ✅ deploy-linux.sh (250 lines)
│   │   ├── ✅ CROSS_PLATFORM_README.md
│   │   ├── ✅ ADAPTABLE_ENGINE_INTEGRATION.md
│   │   ├── ✅ app.py (existing)
│   │   ├── ML/
│   │   │   ├── venv/ (created by deploy)
│   │   │   ├── requirements.txt (existing, pinned)
│   │   │   ├── requirements.lock (created by deploy)
│   │   │   ├── ESI_Engine.py (existing, needs update)
│   │   │   ├── models/
│   │   │   │   └── medgemma-4b-it.gguf (download here)
│   │   │   └── [other files]
│   │   ├── model_cache/ (created by deploy)
│   │   └── logs/ (created by deploy)
│   │
│   └── [frontend, etc]
│
├── ✅ DEPLOYMENT_INDEX.md
├── ✅ QUICK_REFERENCE.md
├── ✅ CROSS_PLATFORM_DEPLOYMENT_COMPLETE.md
├── ✅ IMPLEMENTATION_SUMMARY.md
├── ✅ MANIFEST.md
└── [other project files]
```

---

## Summary Diagram

```
┌──────────────────────────────────────────────────────────────┐
│              ADAPTABLE TRIAGE ENGINE SYSTEM                  │
└──────────────────────────────────────────────────────────────┘

                    YOUR APPLICATION
                    (Flask, ML, etc)
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
         get_config()          get_engine()
                │                   │
                └─────────┬─────────┘
                          ▼
        ┌─────────────────────────────────┐
        │   AdaptableTriageEngine         │
        │                                 │
        │   ✅ Auto-Detects Everything   │
        │   ✅ Self-Optimizing           │
        │   ✅ Exports Configuration     │
        └─────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    Platform        Hardware            Models
    • Windows       • CPU               • GGUF
    • Linux         • GPU               • RAG
    • macOS         • RAM               • Transformers
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                    Configuration
                   (50+ Settings)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    Device          Batch Size         Threads
    (GPU/CPU)       (Optimized)        (Optimized)
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
            Better              Production
            Performance         Ready
```

---

**This architecture enables one codebase to work everywhere with automatic optimization! 🚀**
