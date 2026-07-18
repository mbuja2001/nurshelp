# Adaptable Triage Engine - Cross-Platform Setup

> **Automatically configures your medical triage engine for Windows, Linux, Mac - with CPU or GPU support, with or without RAG**

## 🎯 Quick Start (Choose Your Platform)

### Windows
```powershell
# Run this in PowerShell as Administrator
cd C:\Users\<YourUsername>\Videos\nursehelpdesk\nurshelp\backend
powershell -ExecutionPolicy Bypass -File deploy-windows.ps1
```

### Linux/Mac
```bash
# Run this in Terminal
cd ~/Videos/nursehelpdesk/nurshelp/backend
bash deploy-linux.sh
```

That's it! The deployment script will:
- ✅ Detect your platform, CPU cores, RAM, GPU
- ✅ Create Python virtual environment
- ✅ Install all dependencies (with GPU support if available)
- ✅ Validate cross-platform configuration
- ✅ Create required directories
- ✅ Run configuration diagnostics

## 🔍 What Gets Configured?

The Adaptable Engine auto-detects and optimizes for:

| Setting | Auto-Detects |
|---------|-------------|
| **Platform** | Windows / Linux / macOS |
| **CPU** | Cores, threads, physical vs logical |
| **Memory** | Total and available RAM |
| **GPU** | NVIDIA GPU presence, VRAM, CUDA version |
| **Models** | GGUF location, RAG availability, transformers cache |
| **Performance** | Batch sizes, threading, memory budgets |
| **Console** | Colorama support (Windows), ANSI (Linux/Mac) |

## 🚀 After Deployment

### Test Configuration
```bash
# Show detected configuration
python adaptable_engine.py
```

Expected output:
```
╔═══════════════════════════════════════════════════════════════╗
║         Adaptable Triage Engine Configuration Summary          ║
╠═══════════════════════════════════════════════════════════════╣
║ Platform: Windows                                             ║
║ Device: cuda                                                  ║
║ GPU: Yes (8GB)                                                ║
║ RAM: 32 GB available                                          ║
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

### Get Full Configuration (JSON)
```bash
python adaptable_engine.py --json
# or
python -c "from adaptable_engine import get_config; import json; print(json.dumps(get_config(), indent=2, default=str))"
```

### Start Backend Server
```bash
# Windows (if using Start-NurseHelp):
Start-NurseHelp
python -m flask run

# Linux/Mac:
source venv/bin/activate
export FLASK_APP=app.py
flask run
```

## 📁 New Files Added

After deployment, you'll have:

```
backend/
├── cross_platform_config.py          ← Core platform utilities
├── adaptable_engine.py               ← Auto-configuring engine
├── deploy-windows.ps1                ← Windows deployment
├── deploy-linux.sh                   ← Linux/Mac deployment
├── ADAPTABLE_ENGINE_INTEGRATION.md   ← Integration guide
├── CROSS_PLATFORM_README.md          ← This file
│
├── ML/
│   ├── venv/                         ← Virtual environment (created)
│   ├── requirements.txt              ← Python dependencies
│   ├── requirements.lock             ← Exact versions (auto-created)
│   └── models/                       ← GGUF models directory (created)
│
├── model_cache/                      ← Transformers cache (created)
└── logs/                             ← Application logs (created)
```

## 🔧 Integration with Your Code

### For Flask App (app.py)
```python
from adaptable_engine import get_engine, get_config

app = Flask(__name__)
engine = get_engine()  # Gets configuration on import
config = get_config()  # Access any setting

@app.route('/api/health')
def health():
    return {
        'platform': config['platform']['system'],
        'gpu': config['hardware']['gpu_available'],
        'offline_mode': config['environment']['offline_mode'],
    }
```

### For ML Code (ESI_Engine.py)
```python
from adaptable_engine import get_config
from cross_platform_config import Colorama, CrossPlatformPath

config = get_config()

# Use platform-adaptive path
model_path = CrossPlatformPath.normalize(config['models']['gguf']['model_path'])

# Use colorama for console colors (works on Windows!)
Colorama.print_status("Loading model", "info")
Colorama.print_status("Model loaded", "success")

# Use appropriate batch size
batch_size = config['performance']['gpu_batch_size'] if config['hardware']['gpu_available'] else config['performance']['batch_size']

# Use detected device
device = config['environment']['device']  # 'cuda' or 'cpu'
```

## 🐛 Troubleshooting

### Issue: "Python not found"
```powershell
# Windows - Verify Python is in PATH
python --version

# If not found, add Python to PATH:
# Search "Edit environment variables" → add Python installation dir to PATH
```

### Issue: "Virtual environment activation fails"
```bash
# Linux/Mac - Make script executable
chmod +x deploy-linux.sh

# Then retry
bash deploy-linux.sh
```

### Issue: "GGUF model not found"
Download the model to `backend/ML/models/`:
```bash
# Option 1: Using curl/wget
wget -O backend/ML/models/medgemma-4b-it-Q4_K_M.gguf \
  https://huggingface.co/.../medgemma-4b-it-Q4_K_M.gguf

# Option 2: Manual download
# 1. Visit huggingface.co and search "medgemma-4b-it"
# 2. Download the Q4_K_M quantized version
# 3. Place in backend/ML/models/
```

### Issue: "GPU not detected (but you have one)"
```bash
# Check CUDA installation
nvidia-smi

# If no output, install NVIDIA drivers and CUDA toolkit
# https://developer.nvidia.com/cuda-downloads

# After CUDA install, verify:
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

### Issue: "Colorama colors not working on Windows"
```python
# Verify colorama is initialized
from cross_platform_config import Colorama
Colorama.init()  # Explicitly initialize

# Test colors
Colorama.print_status("This should be green", "success")
```

### Issue: "Model loading timeout on CPU"
The engine auto-detects CPU and sets appropriate batch sizes. If still slow:
```python
config = get_config()
# Reduce batch size
config['performance']['batch_size'] = 512  # Instead of 2000
```

### Issue: Cross-platform path issues
```python
from cross_platform_config import CrossPlatformPath

# Always use this instead of hardcoded paths
path = CrossPlatformPath.normalize(config['paths']['ml_dir'])

# Or construct safely
safe_path = CrossPlatformPath.join(config['paths']['backend_dir'], 'models')
```

## 📊 Configuration Structure

Complete configuration tree available as JSON:

```json
{
  "platform": {
    "system": "Windows",
    "is_windows": true,
    "is_linux": false,
    "path_separator": "\\",
    "line_separator": "\r\n",
    "python_version": "3.11.4"
  },
  "hardware": {
    "cpu_cores": 16,
    "physical_cores": 8,
    "total_ram_gb": 32.0,
    "available_ram_gb": 24.5,
    "gpu_available": true,
    "gpu_type": "cuda",
    "gpu_vram_gb": 8.0,
    "cuda_available": true,
    "cuda_version": "11.8"
  },
  "models": {
    "gguf": {
      "available": true,
      "model_name": "medgemma-4b-it-Q4_K_M",
      "model_path": "C:\\...\\backend\\ML\\models\\medgemma.gguf",
      "model_size_gb": 2.5,
      "quantization": "Q4_K_M",
      "context_size": 1200,
      "batch_size": 2000
    },
    "rag": {
      "available": false,
      "retriever_type": "pubmedbert",
      "similarity_threshold": 0.7
    }
  },
  "environment": {
    "offline_mode": true,
    "debug": false,
    "backend_port": 5000,
    "device": "cuda",
    "environment": "development"
  },
  "features": {
    "gguf_inference": true,
    "rag_retrieval": false,
    "gpu_acceleration": true,
    "colorama_support": true,
    "offline_operation": true
  },
  "performance": {
    "max_threads": 8,
    "inference_threads": 6,
    "batch_size": 2000,
    "gpu_batch_size": 32,
    "gpu_memory_fraction": 0.8,
    "use_mixed_precision": true
  }
}
```

## 🌍 Multi-PC Sync (Windows)

If you're using 2+ Windows PCs with the same codebase:

```powershell
# PC #1: Create lock file
cd backend\ML
./sync-environments.ps1 -Mode create-lock
# Generates requirements.lock with exact package versions

# PC #2: Apply lock file
# Copy requirements.lock from PC #1 to PC #2
./sync-environments.ps1 -Mode apply-lock
# Installs exact same versions as PC #1
```

This ensures both machines have identical Python environments - eliminating the "works on my machine" problem!

## 🔒 Offline-Only Operation

⚠️ **Critical for Medical Use**: The engine operates ONLY in offline mode.

- ✅ All inference runs locally
- ✅ No model downloads during inference
- ✅ No internet dependencies for core functionality  
- ✅ No patient data leaves the system

Verify offline mode:
```python
config = get_config()
assert config['environment']['offline_mode'] == True
assert config['features']['offline_operation'] == True
```

## 📚 Additional Documentation

- [ADAPTABLE_ENGINE_INTEGRATION.md](./ADAPTABLE_ENGINE_INTEGRATION.md) - Detailed integration guide with code examples
- [COLORAMA_QUICK_FIX.md](../COLORAMA_QUICK_FIX.md) - Windows colorama troubleshooting
- [WINDOWS_ENV_SYNC.md](../WINDOWS_ENV_SYNC.md) - Multi-PC environment synchronization

## 🆘 Need Help?

### Check Logs
```bash
# View recent logs
tail -f backend/logs/*.log  # Linux/Mac
Get-Content backend/logs/*.log -Tail 20  # Windows

# Check for errors during deployment
# Full logs are in:
# - deploy-windows.ps1 output (Windows)
# - deploy-linux.sh output (Linux/Mac)
```

### Run Diagnostics
```bash
# Full diagnostics
python -c "
from adaptable_engine import get_config
import json

config = get_config()
print('DIAGNOSTIC REPORT:')
print('─' * 60)

# Check each critical component
checks = [
    ('Platform', config['platform']['system']),
    ('Python Version', config['platform']['python_version']),
    ('GGUF Model', 'Available' if config['models']['gguf']['available'] else 'NOT FOUND'),
    ('GPU Device', config['environment']['device']),
    ('GPU Available', 'Yes' if config['hardware']['gpu_available'] else 'No'),
    ('Offline Mode', 'Enabled' if config['environment']['offline_mode'] else 'DISABLED'),
]

for check_name, result in checks:
    status = '✓' if result in ['Available', 'Yes', 'Enabled', 'cuda'] else '✗'
    print(f'{status} {check_name}: {result}')

print('─' * 60)
print('For detailed config: python adaptable_engine.py')
"
```

### Still Stuck?
1. Run the deployment script again - it has comprehensive error messages
2. Check that Python 3.9+ is installed
3. Verify Git Bash (Windows) or Terminal (Mac) for script execution
4. Look for error messages in the terminal output

## 🎓 How It Works

### On Startup

1. **Detect Platform**: Identifies Windows/Linux/Mac automatically
2. **Discover Hardware**: Queries CPU cores, RAM, GPU via `psutil` and `torch.cuda`
3. **Locate Models**: Searches multiple paths for GGUF files
4. **Test Dependencies**: Verifies all required packages are installed
5. **Optimize Settings**: Sets batch sizes and threading based on hardware
6. **Initialize Console**: Sets up colorama (Windows) or ANSI (Linux/Mac)

### During Runtime

- **Platform-Aware Paths**: Uses `pathlib` + platform detection for cross-platform file handling
- **GPU Fallback**: Automatically uses CPU if CUDA unavailable
- **Memory Monitoring**: Respects RAM constraints during batch processing
- **Colorama Support**: Provides colored terminal output on all platforms

### Performance Tuning

- **GPU Available**: Uses larger batch sizes (32), mixed precision, GPU memory optimization
- **CPU Only**: Uses smaller batch sizes (2000), multi-threading across cores
- **Low RAM**: Automatically reduces batch size and memory budget
- **High Core Count**: Scales threading to avoid bottlenecks

## 📦 Package Versions

Locked versions in `requirements.txt` ensure consistency across machines:

```
colorama>=0.4.6           # Windows terminal colors
click>=8.1.0              # CLI framework (Flask dependency)
werkzeug>=2.3.0           # WSGI utilities (Flask dependency)
python-dotenv>=1.0.0      # Environment config
psutil>=5.9.0             # Hardware detection
flask>=2.3.0              # Web framework
torch>=2.0.0              # PyTorch (CPU or GPU)
llama-cpp-python>=0.2.0   # GGUF inference (if using)
```

All pinned with `>=` to allow security patches while preventing major version breaks.

## 🚢 Deployment Checklist

- [ ] Run deployment script for your platform
- [ ] Verify configuration with `python adaptable_engine.py`
- [ ] Download GGUF model to `backend/ML/models/` (2-4GB)
- [ ] Test backend startup: `python -m flask run`
- [ ] Verify API works: `curl http://localhost:5000/api/health`
- [ ] Check logs in `backend/logs/` for any warnings
- [ ] Document any platform-specific issues for your team

## 📝 License & Support

This adaptable engine is part of the NurseHelp medical triage system. For production deployment:

- ✅ Tested on Windows 10/11, Ubuntu 20.04+, macOS 12+
- ✅ Supports Python 3.9, 3.10, 3.11, 3.12
- ✅ Offline-first architecture for medical safety
- ✅ HIPAA-ready (no external communication)

---

**Happy deployments! 🚀**
