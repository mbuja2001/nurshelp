# ⚡ Cross-Platform Engine - Quick Reference Card

## 🚀 Deploy (5 minutes)

### Windows
```powershell
cd C:\Users\<You>\Videos\nursehelpdesk\nurshelp\backend
powershell -ExecutionPolicy Bypass -File deploy-windows.ps1
```

### Linux
```bash
cd ~/Videos/nursehelpdesk/nurshelp/backend
bash deploy-linux.sh
```

✅ Done! Check your configuration:
```bash
python adaptable_engine.py
```

---

## 💻 Use in Code

### Get Configuration
```python
from adaptable_engine import get_config
config = get_config()
```

### Key Settings
```python
# Hardware
config['hardware']['cpu_cores']        # CPU cores
config['hardware']['gpu_available']    # True/False
config['hardware']['available_ram_gb'] # RAM

# Models
config['models']['gguf']['available']      # True/False
config['models']['gguf']['model_path']     # Full path

# Performance
config['performance']['batch_size']         # CPU batch
config['performance']['gpu_batch_size']     # GPU batch
config['performance']['inference_threads']  # Thread count
config['environment']['device']             # 'cuda' or 'cpu'

# Paths
config['paths']['ml_dir']        # ML directory
config['paths']['model_paths']   # Where to find models
config['paths']['logs_dir']      # Logs directory
```

### Smart Inference Code
```python
from adaptable_engine import get_config

config = get_config()

# Hardware-aware batch size
if config['hardware']['gpu_available']:
    batch_size = config['performance']['gpu_batch_size']  # 32
else:
    batch_size = config['performance']['batch_size']      # 2000

# Load with right device
device = config['environment']['device']  # 'cuda' or 'cpu'

# Use model path
model = load_model(config['models']['gguf']['model_path'])
```

### Platform-Safe Paths
```python
from cross_platform_config import CrossPlatformPath
from adaptable_engine import get_config

config = get_config()

# Join paths safely (Windows & Linux)
model_dir = CrossPlatformPath.join(
    config['paths']['ml_dir'], 
    'models'
)

# Normalize path for current OS
safe_path = CrossPlatformPath.normalize(any_path)

# Create directories
CrossPlatformPath.ensure_dir(config['paths']['logs_dir'])
```

### Colored Console Output
```python
from cross_platform_config import Colorama

Colorama.print_status("Starting triage", "info")      # Blue
Colorama.print_status("Success!", "success")          # Green
Colorama.print_status("Check GPU", "warning")        # Yellow
Colorama.print_status("Failed", "error")             # Red
```

---

## 📊 Check Configuration

### See Auto-Detected Settings
```bash
python adaptable_engine.py
```

### Export as JSON
```bash
python -c "
from adaptable_engine import get_config
import json
print(json.dumps(get_config(), indent=2, default=str))
"
```

### Diagnose Issues
```bash
python -c "
from adaptable_engine import get_config
c = get_config()

print('Platform:', c['platform']['system'])
print('GPU:', 'Yes' if c['hardware']['gpu_available'] else 'No')
print('GGUF:', 'Found' if c['models']['gguf']['available'] else 'NOT FOUND')
print('Device:', c['environment']['device'])
print('Offline:', c['environment']['offline_mode'])
"
```

---

## 🐛 Quick Fixes

### GGUF Model Not Found
```bash
# Download to:
# backend/ML/models/medgemma-4b-it-Q4_K_M.gguf
# (2-4 GB file)
```

### GPU Not Detected (but you have one)
```bash
# Windows: Install NVIDIA drivers
nvidia-smi

# Linux: Install CUDA toolkit
nvidia-smi

# Python: Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Python Version Wrong
```bash
# Need 3.9+
python --version

# If too old, download Python 3.11 from python.org
```

### Colorama Colors Not Working
```python
from cross_platform_config import Colorama
Colorama.init()  # Initialize explicitly
```

### Virtual Environment Broken
```bash
# Windows:
rm -r ML\venv
powershell -ExecutionPolicy Bypass -File deploy-windows.ps1

# Linux:
rm -r ML/venv
bash deploy-linux.sh
```

---

## 📁 Files Added

```
backend/
├── cross_platform_config.py        ← Platform utilities
├── adaptable_engine.py             ← Auto-config engine
├── deploy-windows.ps1              ← Windows setup
├── deploy-linux.sh                 ← Linux setup
├── CROSS_PLATFORM_README.md        ← Full guide
├── ADAPTABLE_ENGINE_INTEGRATION.md ← Integration
│
└── ML/
    ├── venv/                       ← Virtual environment
    ├── requirements.lock           ← Exact versions
    └── models/                     ← Put GGUF here
```

---

## 🎯 Integration Checklist

- [ ] Run deployment script (Windows or Linux)
- [ ] Download GGUF model to `backend/ML/models/`
- [ ] Test with: `python adaptable_engine.py`
- [ ] Update ESI_Engine.py:
  ```python
  from adaptable_engine import get_config
  config = get_config()
  # Use config for paths and settings
  ```
- [ ] Replace hardcoded paths with config values
- [ ] Test backend: `python -m flask run`
- [ ] Check: `curl http://localhost:5000/api/health`

---

## 🔧 Common Patterns

### Hardware-Aware Loop
```python
config = get_config()
max_iter = 1000 if config['hardware']['gpu_available'] else 100
for i in range(max_iter):
    process_batch()
```

### Device Selection
```python
device = config['environment']['device']
model = Model().to(device)
```

### Multi-Platform Path
```python
base = config['paths']['backend_dir']
model_path = CrossPlatformPath.join(base, 'models', 'file.gguf')
```

### Offline Verification
```python
assert config['environment']['offline_mode'], "Must be offline!"
```

### Performance Tuning
```python
threads = config['performance']['inference_threads']
batch = config['performance']['batch_size']
memory = config['performance']['memory_budget_pct']
```

---

## 📞 Support

| Problem | Fix |
|---------|-----|
| Can't find python | Install Python 3.9+ |
| GGUF model missing | Download to `backend/ML/models/` |
| GPU not detected | Check NVIDIA drivers + CUDA |
| Colors not working | Run: `Colorama.init()` |
| 2 PCs out of sync | See: `WINDOWS_ENV_SYNC.md` |
| Path issues | Use: `CrossPlatformPath` |
| Wrong batch sizes | Delete `venv/`, rerun deployment |

---

## 📚 Full Docs

- **Getting Started**: [CROSS_PLATFORM_README.md](./nurshelp/backend/CROSS_PLATFORM_README.md)
- **Integration**: [ADAPTABLE_ENGINE_INTEGRATION.md](./nurshelp/backend/ADAPTABLE_ENGINE_INTEGRATION.md)
- **Windows Sync**: [WINDOWS_ENV_SYNC.md](./WINDOWS_ENV_SYNC.md)
- **Index**: [DEPLOYMENT_INDEX.md](./DEPLOYMENT_INDEX.md)

---

**Deploy. Configure. Adapt. 🚀**
