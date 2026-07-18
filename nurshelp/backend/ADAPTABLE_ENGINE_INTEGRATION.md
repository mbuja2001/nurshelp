# Adaptable Triage Engine Integration Guide

## Overview

The **Adaptable Triage Engine** automatically configures itself for your system:
- **Platform Detection**: Windows, Linux, Mac auto-detection
- **Hardware Discovery**: CPU cores, RAM, GPU availability
- **Model Resolution**: Auto-finds GGUF models, validates RAG setup
- **Performance Tuning**: Adjusts batch sizes, threading based on hardware
- **Colorama Support**: Cross-platform terminal colors (Windows-friendly)

## Quick Start

### 1. Run Adaptable Engine Diagnostics

```bash
# From backend directory
python adaptable_engine.py
```

**Output Example (Windows):**
```
╔═══════════════════════════════════════════════════════════════╗
║         Adaptable Triage Engine Configuration Summary          ║
╠═══════════════════════════════════════════════════════════════╣
║ Platform: Windows                                             ║
║ Device: cuda                                                  ║
║ GPU: Yes (8GB)                                                ║
║ RAM: 32 GB available                                          ║
║ Offline Mode: Enabled                                         ║
╠═══════════════════════════════════════════════════════════════╣
║ Models Available:                                             ║
║   • GGUF: ✅                                                  ║
║   • RAG: ❌                                                   ║
║ Features Enabled:                                             ║
║   • GPU Acceleration: ✅                                      ║
║   • GGUF Inference: ✅                                        ║
║   • RAG Retrieval: ❌                                         ║
╚═══════════════════════════════════════════════════════════════╝
```

### 2. Use in Your Flask App

```python
from adaptable_engine import get_engine, get_config

app = Flask(__name__)

# Initialize engine on startup
engine = get_engine()
config = get_config()

# Access platform-specific settings
@app.route('/api/health')
def health():
    return {
        'platform': config['platform']['system'],
        'gpu_available': config['hardware']['gpu_available'],
        'offline_mode': config['environment']['offline_mode'],
    }
```

### 3. Use in Your ML Code (ESI_Engine.py)

```python
from adaptable_engine import get_engine, get_config
from cross_platform_config import Colorama, CrossPlatformPath

# Get config
config = get_config()

# Platform-adaptive path handling
model_path = CrossPlatformPath.normalize(config['models']['gguf']['model_path'])

# Platform-aware logging
Colorama.print_status("Loading model", "info")

# Hardware-aware tuning
batch_size = config['performance']['gpu_batch_size'] if config['hardware']['gpu_available'] else config['performance']['batch_size']

# Device selection (auto-detected)
device = config['environment']['device']  # 'cuda' or 'cpu'
```

## Configuration Structure

### Platform Settings
```python
config['platform'] = {
    'system': 'Windows' | 'Linux' | 'Darwin',
    'is_windows': bool,
    'is_linux': bool,
    'path_separator': '\\' | '/',
    'line_separator': '\r\n' | '\n',
}
```

### Hardware Detection
```python
config['hardware'] = {
    'cpu_cores': int,           # Logical cores
    'physical_cores': int,      # Physical cores
    'total_ram_gb': float,      # Total RAM
    'available_ram_gb': float,  # Available RAM
    'gpu_available': bool,
    'gpu_type': 'none' | 'cuda' | 'rocm',
    'gpu_vram_gb': float,
    'cuda_available': bool,
    'cuda_version': str,
}
```

### Model Configuration
```python
config['models'] = {
    'gguf': {
        'available': bool,
        'model_name': str,
        'model_path': str,
        'model_size_gb': float,
        'quantization': str,      # e.g. 'Q4_K_M'
        'context_size': int,
        'batch_size': int,
    },
    'rag': {
        'available': bool,
        'retriever_type': str,
        'index_path': str,
        'similarity_threshold': float,
    },
    'transformers': {
        'available': bool,
        'cache_location': str,
    },
}
```

### Environment Settings
```python
config['environment'] = {
    'offline_mode': True,           # Always offline for medical use
    'debug': bool,
    'backend_port': int,
    'device': 'cuda' | 'cpu',       # Auto-detected
    'disable_rag': bool,
    'force_pure_llm': bool,
    'environment': 'development' | 'production',
}
```

### Feature Flags
```python
config['features'] = {
    'gguf_inference': bool,         # GGUF model available
    'rag_retrieval': bool,          # RAG available
    'gpu_acceleration': bool,       # GPU available
    'colorama_support': bool,       # Terminal colors
    'offline_operation': bool,      # Always True
}
```

### Performance Tuning
```python
config['performance'] = {
    'max_threads': int,
    'inference_threads': int,
    'batch_size': int,
    'context_size': int,
    'use_memory_mapping': bool,
    'memory_budget_pct': int,
    'gpu_batch_size': int,          # GPU only
    'gpu_memory_fraction': float,   # GPU only
    'use_mixed_precision': bool,    # GPU only
}
```

## Common Usage Patterns

### Pattern 1: Hardware-Aware Inference

```python
from adaptable_engine import get_config

config = get_config()

# Use GPU batch size on GPU, CPU batch size on CPU
if config['hardware']['gpu_available']:
    batch_size = config['performance']['gpu_batch_size']
    inference_threads = 2  # GPU handles its own threading
else:
    batch_size = config['performance']['batch_size']
    inference_threads = config['performance']['inference_threads']

# Load model with appropriate settings
model = LlamaForCausalLM.from_pretrained(
    config['models']['gguf']['model_path'],
    device_map=config['environment']['device'],
    batch_size=batch_size,
    n_threads=inference_threads,
    use_mlock=config['performance']['use_memory_mapping'],
)
```

### Pattern 2: Cross-Platform Path Handling

```python
from cross_platform_config import CrossPlatformPath
from adaptable_engine import get_config

config = get_config()

# Auto-detect and normalize paths
model_dir = CrossPlatformPath.normalize(config['paths']['ml_dir'])
logs_dir = CrossPlatformPath.normalize(config['paths']['logs_dir'])

# Create directories if needed
CrossPlatformPath.ensure_dir(logs_dir)
```

### Pattern 3: Platform-Aware Logging

```python
from cross_platform_config import Colorama
from adaptable_engine import get_config

config = get_config()

if config['platform']['is_windows']:
    # Windows: use colorama for colored output
    Colorama.print_status("Processing triage", "info")
    Colorama.print_status("Model loaded", "success")
else:
    # Linux: native ANSI support
    Colorama.print_status("Processing triage", "info")
```

### Pattern 4: Feature-Based Initialization

```python
from adaptable_engine import get_config

config = get_config()

# Initialize only available features
if config['features']['gguf_inference']:
    gguf_engine = initialize_gguf_model()

if config['features']['rag_retrieval']:
    rag_engine = initialize_rag_system()

if config['features']['gpu_acceleration']:
    enable_cuda_optimization()
```

### Pattern 5: Offline-Only Operation

```python
from adaptable_engine import get_config

config = get_config()

# Ensure offline operation
assert config['environment']['offline_mode'] == True, "Offline mode must be enabled for medical use"
assert config['features']['gpu_acceleration'] or config['performance']['max_threads'] > 0, "No compute available"

# All inference runs completely offline
triage_result = engine.infer(patient_data)
```

## Migration Path from ESI_Engine.py

### Step 1: Add Import
```python
from adaptable_engine import get_engine, get_config
from cross_platform_config import Colorama, CrossPlatformPath
```

### Step 2: Get Configuration in `__init__`
```python
class ESI_Engine:
    def __init__(self):
        self.config = get_config()
        self.device = self.config['environment']['device']
        self.threads = self.config['performance']['inference_threads']
```

### Step 3: Use Configuration for Model Loading
```python
def load_model(self):
    model_path = self.config['models']['gguf']['model_path']
    batch_size = self.config['performance']['batch_size']
    
    self.llm = Llama(
        model_path=model_path,
        n_threads=self.threads,
        n_batch=batch_size,
        # ... other settings
    )
```

### Step 4: Replace Hard-Coded Paths
```python
# Before:
model_path = "C:\\models\\medgemma.gguf"

# After:
model_path = self.config['models']['gguf']['model_path']
```

### Step 5: Use Colorama for Logging
```python
# Before:
print("Model loaded")

# After:
Colorama.print_status("Model loaded", "success")
```

## Diagnostics & Troubleshooting

### Check Configuration
```bash
python adaptable_engine.py
```

### Verbose Configuration Output
```python
import json
from adaptable_engine import get_config

config = get_config()
print(json.dumps(config, indent=2, default=str))
```

### Common Issues

**Issue: GGUF model not found**
```python
config = get_config()
if not config['models']['gguf']['available']:
    print("GGUF model not found in:")
    for path in config['paths']['model_paths']:
        print(f"  - {path}")
```

**Issue: GPU not detected**
```python
config = get_config()
if not config['hardware']['gpu_available']:
    print(f"GPU not available. Using CPU with {config['hardware']['physical_cores']} cores")
```

**Issue: Colorama not working (Windows)**
```python
from cross_platform_config import Colorama
Colorama.disable_if_needed()
```

## Environment Variables (Optional Overrides)

```bash
# Force CPU-only
export FORCE_CPU=1

# Disable RAG
export DISABLE_RAG=1

# Force pure LLM (no RAG)
export FORCE_PURE_LLM=1

# Set backend port
export BACKEND_PORT=5001

# Enable debug mode
export DEBUG=1

# Set environment
export ENVIRONMENT=production
```

## Files Structure

```
backend/
├── cross_platform_config.py      # Core platform utilities
├── adaptable_engine.py           # Engine configuration & auto-setup
├── ML/
│   ├── ESI_Engine.py             # (Update to use adaptable_engine)
│   ├── requirements.txt          # Pinned dependencies (Windows-safe)
│   └── models/
│       └── medgemma-4b-it.gguf   # (auto-detected)
└── model_cache/                  # (auto-created)
```

## Next Steps

1. ✅ Run `python adaptable_engine.py` to see your configuration
2. ✅ Review configuration summary for any issues
3. ✅ Update `ESI_Engine.py` to use `get_config()` 
4. ✅ Replace hard-coded paths with platform-adaptive ones
5. ✅ Test on both Windows and Linux with same codebase

## Questions?

- **Platform path issues?** → Check `cross_platform_config.CrossPlatformPath`
- **Terminal color issues?** → Check `cross_platform_config.Colorama`
- **Configuration not auto-detecting?** → Run `python adaptable_engine.py` for diagnostics
- **Model not found?** → Check paths in configuration under `paths.model_paths`
- **GPU not detected?** → Verify CUDA installation with `python -c "import torch; print(torch.cuda.is_available())"`
