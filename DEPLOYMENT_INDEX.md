# 🏥 NurseHelp Cross-Platform Deployment - Complete Index

> **One codebase. Multiple platforms. Auto-configured performance.**

## 📚 Documentation Files

Start here based on your situation:

### 🚀 Getting Started (Pick One)
- **[CROSS_PLATFORM_README.md](./CROSS_PLATFORM_README.md)** ← **START HERE** for initial setup
  - Quick start for Windows & Linux
  - Post-deployment verification
  - Troubleshooting common issues
  - Configuration overview

### 🔧 Integration & Development
- **[ADAPTABLE_ENGINE_INTEGRATION.md](./ADAPTABLE_ENGINE_INTEGRATION.md)** - Deep dive integration guide
  - How to use adaptable_engine in your code
  - Code patterns and best practices
  - Hardware-aware inference examples
  - Cross-platform path handling
  - Migration path from hardcoded configs

### 🪟 Windows-Specific
- **[../COLORAMA_QUICK_FIX.md](../COLORAMA_QUICK_FIX.md)** - Windows terminal color fix
  - One-time setup commands
  - Quick verification checklist
  - Nuclear reset option
  
- **[../WINDOWS_ENV_SYNC.md](../WINDOWS_ENV_SYNC.md)** - Multi-PC environment sync
  - Sync Python environments between 2+ Windows machines
  - Lock file mechanism explained
  - Detailed troubleshooting for cross-PC issues

## 💻 Python Files

### Core Platform Abstraction
- **`cross_platform_config.py`** - Low-level platform utilities
  - OS detection (Windows/Linux/Mac)
  - Path normalization
  - Console color support (Colorama wrapper)
  - Environment variable handling
  - GGUF model path resolution
  
  **Usage:** 
  ```python
  from cross_platform_config import (
      Colorama, CrossPlatformPath, 
      IS_WINDOWS, PLATFORM
  )
  ```

### Adaptable Engine Configuration
- **`adaptable_engine.py`** - Auto-configuring engine
  - Hardware detection (CPU, RAM, GPU)
  - Model discovery (GGUF, RAG, transformers)
  - Performance tuning based on hardware
  - Feature flags
  - Configuration export as JSON
  
  **Usage:**
  ```python
  from adaptable_engine import get_engine, get_config
  
  engine = get_engine()
  config = get_config()
  ```

## 🚀 Deployment Scripts

### Windows Deployment
- **`deploy-windows.ps1`** - Full Windows setup automation
  - Detects hardware (CPU, RAM, GPU, NVIDIA)
  - Creates Python virtual environment
  - Installs all dependencies (with GPU support)
  - Validates configuration
  - Creates shortcuts for quick access
  
  **Run:** `powershell -ExecutionPolicy Bypass -File deploy-windows.ps1`

### Linux/Mac Deployment  
- **`deploy-linux.sh`** - Full Linux/Mac setup automation
  - Detects system configuration
  - Creates Python virtual environment
  - Installs dependencies
  - Tests CUDA if available
  - Validates all components
  
  **Run:** `bash deploy-linux.sh`

## 📊 Configuration & Reference

### Auto-Detected Configuration Areas
The adaptable engine detects and optimizes:

| Category | Details |
|----------|---------|
| **Platform** | OS type, path separators, line endings |
| **Hardware** | CPU cores, RAM, GPU presence, VRAM |
| **Models** | GGUF location, RAG setup, transformers cache |
| **Environment** | Offline mode, debug flag, backend port, device |
| **Features** | What's available (GGUF, RAG, GPU) |
| **Performance** | Batch sizes, threading, memory budgets |

### Access Configuration
```python
# Full config as Python dict
config = get_config()

# By category
config['platform']       # OS info
config['hardware']       # CPU, GPU, RAM
config['models']         # Available models
config['environment']    # Runtime settings
config['features']       # Feature flags
config['performance']    # Tuning parameters
config['paths']          # All directories

# Print readable summary
from adaptable_engine import print_config_summary
print_config_summary()
```

## 🎯 Workflow by Role

### 🚀 DevOps / System Administrator
1. Read: [CROSS_PLATFORM_README.md](./CROSS_PLATFORM_README.md) "Quick Start"
2. Run: `deploy-windows.ps1` (Windows) or `deploy-linux.sh` (Linux)
3. Verify: `python adaptable_engine.py`
4. If issues: [Troubleshooting](./CROSS_PLATFORM_README.md#-troubleshooting) section

### 👨‍💻 Backend Developer
1. Read: [ADAPTABLE_ENGINE_INTEGRATION.md](./ADAPTABLE_ENGINE_INTEGRATION.md)
2. Study: Common usage patterns (hardware-aware inference, cross-platform paths)
3. Update `ESI_Engine.py` to use `get_config()`
4. Test: `python adaptable_engine.py` to verify your environment
5. Reference: Configuration structure docs

### 📚 ML/Data Scientist
1. Quick read: [ADAPTABLE_ENGINE_INTEGRATION.md](./ADAPTABLE_ENGINE_INTEGRATION.md) "Pattern 1: Hardware-Aware Inference"
2. Use: `config['hardware']['gpu_available']` to optimize batch sizes
3. Use: `config['performance']` for tuning parameters
4. Test: Different hardware configurations (GPU vs CPU)

### 🐛 Troubleshooter
1. Run: `python adaptable_engine.py` to see detected configuration
2. Check: [Troubleshooting](./CROSS_PLATFORM_README.md#-troubleshooting) in README
3. For Windows: [WINDOWS_ENV_SYNC.md](../WINDOWS_ENV_SYNC.md) multi-PC issues
4. For colors: [COLORAMA_QUICK_FIX.md](../COLORAMA_QUICK_FIX.md)
5. Run: Diagnostics script (see below)

## 🆘 Diagnostic Tools

### 1. Basic Configuration Check
```bash
python adaptable_engine.py
```
Shows detected platform, hardware, models, features in readable format.

### 2. Detailed JSON Configuration
```bash
python -c "
from adaptable_engine import get_config
import json
print(json.dumps(get_config(), indent=2, default=str))
"
```
Complete configuration as JSON for detailed inspection.

### 3. Dependency Verification
```bash
python -c "
import sys
deps = ['pathlib', 'platform', 'os', 'sys', 'psutil', 'colorama', 'flask', 'click', 'werkzeug', 'python_dotenv']
for dep in deps:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError:
        print(f'✗ {dep} MISSING')
        sys.exit(1)
"
```
Verify all required packages are installed.

### 4. GPU Detection
```bash
python -c "
from adaptable_engine import get_config
c = get_config()
print(f'GPU Available: {c[\"hardware\"][\"gpu_available\"]}')
print(f'Device: {c[\"environment\"][\"device\"]}')
if c['hardware']['gpu_available']:
    print(f'GPU Type: {c[\"hardware\"][\"gpu_type\"]}')
    print(f'VRAM: {c[\"hardware\"][\"gpu_vram_gb\"]}GB')
    print(f'CUDA: {c[\"hardware\"][\"cuda_version\"]}')
"
```
Check GPU detection and CUDA setup.

### 5. Model Discovery
```bash
python -c "
from adaptable_engine import get_config
c = get_config()
gguf = c['models']['gguf']
print(f'GGUF Available: {gguf[\"available\"]}')
if gguf['available']:
    print(f'Path: {gguf[\"model_path\"]}')
    print(f'Size: {gguf[\"model_size_gb\"]}GB')
    print(f'Quantization: {gguf[\"quantization\"]}')
"
```
Check GGUF model detection.

## 📁 File Structure After Deployment

```
backend/
├── cross_platform_config.py          ✅ Core platform utilities
├── adaptable_engine.py               ✅ Auto-configuring engine
├── deploy-windows.ps1                ✅ Windows deployment script
├── deploy-linux.sh                   ✅ Linux/Mac deployment script
├── CROSS_PLATFORM_README.md          ✅ Getting started guide
├── ADAPTABLE_ENGINE_INTEGRATION.md   ✅ Integration guide
│
├── ML/
│   ├── venv/                         📦 Virtual environment (created by deploy)
│   ├── requirements.txt              📋 Pinned dependencies
│   ├── requirements.lock             🔒 Exact versions (created by deploy)
│   ├── ESI_Engine.py                 🔄 (Should integrate adaptable_engine)
│   ├── models/                       📂 GGUF models (created by deploy)
│   │   └── medgemma-4b-it.gguf       ⬇️ Download separately
│   └── [other ML files]
│
├── model_cache/                      📂 Transformers cache (created by deploy)
├── logs/                             📂 Application logs (created by deploy)
│
└── [other backend files]

../
├── COLORAMA_QUICK_FIX.md             🪟 Windows colorama fix
├── WINDOWS_ENV_SYNC.md               🔄 Multi-PC sync guide
└── [other project files]
```

## ✅ Deployment Checklist

- [ ] Choose your platform (Windows / Linux / Mac)
- [ ] Run appropriate deployment script
- [ ] Verify output says "✓ Adaptable Triage Engine Deployment Complete!"
- [ ] Run `python adaptable_engine.py` to see configuration
- [ ] Download GGUF model to `backend/ML/models/` (if not auto-found)
- [ ] Update `ESI_Engine.py` to import from `adaptable_engine`
- [ ] Test backend: `python -m flask run`
- [ ] Check logs in `backend/logs/` for warnings

## 🔄 Migration Path (If Upgrading Existing Setup)

### From Hardcoded Paths
```python
# Before:
MODEL_PATH = "C:\\Users\\user\\models\\medgemma.gguf"

# After:
from adaptable_engine import get_config
config = get_config()
MODEL_PATH = config['models']['gguf']['model_path']
```

### From Hardcoded Device
```python
# Before:
device = "cuda" if torch.cuda.is_available() else "cpu"

# After:
from adaptable_engine import get_config
device = get_config()['environment']['device']
```

### From Manual Batch Sizing
```python
# Before:
batch_size = 2000  # Same for all machines

# After:
from adaptable_engine import get_config
config = get_config()
batch_size = config['performance']['gpu_batch_size'] if config['hardware']['gpu_available'] else config['performance']['batch_size']
```

## 📞 Support Matrix

| Issue | Solution |
|-------|----------|
| "Python not found" | Run deployment script - it checks this first |
| "GGUF model not found" | Download manually to `backend/ML/models/` |
| "GPU not detected" | Check NVIDIA drivers + CUDA toolkit installed |
| "Colorama not working (Windows)" | Verify `colorama>=0.4.6` installed; run `Colorama.init()` |
| "Cross-platform paths failing" | Use `CrossPlatformPath` utilities instead of hardcoded paths |
| "2 PCs out of sync (Windows)" | Use `sync-environments.ps1` from `WINDOWS_ENV_SYNC.md` |
| "Wrong batch sizes for hardware" | `get_config()['performance']` is auto-optimized |
| "Offline mode disabled accidentally" | Check `config['environment']['offline_mode']` in code |

## 🎓 Key Concepts

### 1. Adaptable Initialization
The engine self-configures on import - no manual setup needed:
```python
from adaptable_engine import get_engine  # Auto-detects everything
engine = get_engine()                    # Configuration complete
```

### 2. Platform-Agnostic Code
Write once, deploy anywhere:
```python
from cross_platform_config import CrossPlatformPath
path = CrossPlatformPath.normalize(config['paths']['ml_dir'])
# Works on Windows, Linux, Mac automatically
```

### 3. Hardware-Aware Performance
Automatically optimized for your hardware:
```python
config = get_config()
if config['hardware']['gpu_available']:
    # Use GPU-optimized settings
    batch_size = 32
else:
    # Use CPU-optimized settings
    batch_size = 2000
```

### 4. Offline-First Architecture
Always offline, never communicates externally:
```python
assert config['environment']['offline_mode'] == True
# All inference runs completely locally
# HIPAA-compliant, patient data never leaves system
```

## 🚀 Next Steps After Deployment

1. **Verify Setup**
   ```bash
   python adaptable_engine.py
   ```

2. **Download GGUF Model**
   - Download to `backend/ML/models/`
   - Should be 2-4GB depending on quantization

3. **Update ESI_Engine.py**
   - Import: `from adaptable_engine import get_config`
   - Replace hardcoded paths with config values
   - Use config['environment']['device'] for device selection

4. **Test Integration**
   ```bash
   cd backend
   python -m flask run
   curl http://localhost:5000/api/health
   ```

5. **Review Logs**
   ```bash
   tail -f backend/logs/*.log
   ```

## 📖 Quick Reference

### Common Configuration Accesses
```python
from adaptable_engine import get_config

config = get_config()

# Most-used settings:
is_gpu_available = config['hardware']['gpu_available']
device = config['environment']['device']  # 'cuda' or 'cpu'
batch_size = config['performance']['gpu_batch_size'] if is_gpu_available else config['performance']['batch_size']
model_path = config['models']['gguf']['model_path']
offline_mode = config['environment']['offline_mode']  # Always True
cores = config['hardware']['physical_cores']
threads = config['performance']['inference_threads']
```

### Cross-Platform Path Usage
```python
from cross_platform_config import CrossPlatformPath

# Join paths safely
model_dir = CrossPlatformPath.join(config['paths']['ml_dir'], 'models')

# Normalize path for current OS
safe_path = CrossPlatformPath.normalize(some_path)

# Ensure directory exists
CrossPlatformPath.ensure_dir(config['paths']['logs_dir'])
```

### Console Output (Works on All Platforms)
```python
from cross_platform_config import Colorama

Colorama.print_status("Processing triage", "info")      # Blue
Colorama.print_status("Triage complete", "success")    # Green
Colorama.print_status("Using CPU (slow)", "warning")   # Yellow
Colorama.print_status("Failed to load model", "error") # Red
```

## 🎯 Success Criteria

After deployment, you should be able to:

✅ Run `python adaptable_engine.py` and see your configuration  
✅ See "GPU: Yes" if you have NVIDIA GPU, "GPU: No" if CPU-only  
✅ See "GGUF: ✅" once you download the model  
✅ Import `from adaptable_engine import get_config` in your code  
✅ Run backend with `python -m flask run`  
✅ Access API at `http://localhost:5000`  
✅ See colored terminal output (all platforms)  
✅ Verify offline mode is enabled  

---

**You're ready to deploy! 🚀**

Start with: [CROSS_PLATFORM_README.md](./CROSS_PLATFORM_README.md)
