# 🚀 GET STARTED IN 5 MINUTES

> Everything you need to know to deploy and use the Adaptable Triage Engine

---

## ⚡ Quick Deploy (Choose One)

### Windows
```powershell
cd C:\Users\<YourUsername>\Videos\nursehelpdesk\nurshelp\backend
powershell -ExecutionPolicy Bypass -File deploy-windows.ps1
```

### Linux/Mac
```bash
cd ~/Videos/nursehelpdesk/nurshelp/backend
bash deploy-linux.sh
```

✅ **Done!** Script handles everything:
- Detects OS and hardware
- Creates virtual environment
- Installs dependencies
- Validates configuration

---

## ✨ Verify It Works

```bash
python adaptable_engine.py
```

You'll see something like:
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

---

## 📥 Download GGUF Model (Once)

If you see "GGUF: ❌", download the model:

1. Visit: https://huggingface.co/
2. Search: "medgemma-4b-it"
3. Download: `medgemma-4b-it-Q4_K_M.gguf` (2-4GB)
4. Save to: `backend/ML/models/medgemma-4b-it-Q4_K_M.gguf`

Then verify:
```bash
python adaptable_engine.py
# Should now show: GGUF: ✅
```

---

## 💻 Use in Your Code (3 Steps)

### Step 1: Import
```python
from adaptable_engine import get_config
```

### Step 2: Get Configuration
```python
config = get_config()
```

### Step 3: Use Settings
```python
# Your hardware setup
device = config['environment']['device']  # 'cuda' or 'cpu'
batch_size = config['performance']['batch_size']
model_path = config['models']['gguf']['model_path']
threads = config['performance']['inference_threads']

# Load your model with these settings
model = load_model(
    model_path=model_path,
    device=device,
    batch_size=batch_size,
    n_threads=threads,
)
```

---

## 📊 Common Settings

```python
config = get_config()

# Platform Info
config['platform']['system']              # 'Windows', 'Linux', 'Darwin'
config['platform']['is_windows']          # True/False
config['platform']['python_version']      # '3.11'

# Hardware
config['hardware']['cpu_cores']           # 16
config['hardware']['physical_cores']      # 8  
config['hardware']['available_ram_gb']    # 24.5
config['hardware']['gpu_available']       # True/False
config['hardware']['gpu_vram_gb']         # 8.0

# Device & Performance
config['environment']['device']           # 'cuda' or 'cpu'
config['performance']['batch_size']       # 2000 (CPU) or 32 (GPU)
config['performance']['inference_threads'] # 6

# Models & Paths
config['models']['gguf']['model_path']   # Full path to GGUF
config['models']['gguf']['available']    # True/False
config['paths']['ml_dir']                # ML directory path
config['paths']['logs_dir']              # Logs directory path

# Features
config['features']['gpu_acceleration']   # True/False
config['features']['gguf_inference']     # True/False
config['features']['offline_operation']  # Always True
```

---

## 🔄 Example: Smart Batch Loading

```python
from adaptable_engine import get_config

config = get_config()

# Different batch sizes for GPU vs CPU
if config['hardware']['gpu_available']:
    batch_size = 32  # GPU can handle small, fast batches
else:
    batch_size = 2000  # CPU needs larger batches for efficiency

# Load your data in batches
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    results = process_batch(batch, device=config['environment']['device'])
```

---

## 🛠️ Troubleshooting (Quick Fixes)

### Problem: "Python not found"
```bash
# Check Python is installed
python --version

# If not found:
# Download Python 3.9+ from python.org
```

### Problem: "GGUF model not found"
```bash
# Download to correct location:
# backend/ML/models/medgemma-4b-it-Q4_K_M.gguf

# Then verify:
python adaptable_engine.py
```

### Problem: "GPU not detected (but I have one)"
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA in Python
python -c "import torch; print(torch.cuda.is_available())"

# If False: Install NVIDIA CUDA toolkit
```

### Problem: "Virtual environment broken"
```bash
# Delete and recreate
rm -r backend/ML/venv

# Windows:
powershell -ExecutionPolicy Bypass -File deploy-windows.ps1

# Linux:
bash deploy-linux.sh
```

---

## 📁 What Was Created

```
backend/
├── cross_platform_config.py    ← Platform utilities
├── adaptable_engine.py          ← Configuration engine
├── deploy-windows.ps1           ← Windows setup
├── deploy-linux.sh              ← Linux setup
│
├── ML/
│   ├── venv/                    ← Virtual environment
│   ├── requirements.lock        ← Exact versions
│   └── models/
│       └── medgemma-4b-it.gguf ← Download this
│
├── model_cache/                 ← Transformers cache
└── logs/                        ← Application logs
```

---

## ✅ You're Ready!

After these 5 steps, you have:
- ✅ Deployed the adaptable engine
- ✅ Verified it works
- ✅ Downloaded the GGUF model
- ✅ Know how to use `get_config()`
- ✅ Have working examples

---

## 📚 More Information

- **Full Setup Guide**: [CROSS_PLATFORM_README.md](./nurshelp/backend/CROSS_PLATFORM_README.md)
- **Integration Patterns**: [ADAPTABLE_ENGINE_INTEGRATION.md](./nurshelp/backend/ADAPTABLE_ENGINE_INTEGRATION.md)
- **Quick Reference**: [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
- **Complete Index**: [DEPLOYMENT_INDEX.md](./DEPLOYMENT_INDEX.md)
- **Architecture**: [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 🎯 Next Steps

**Option 1: Test Backend Server**
```bash
cd backend
python -m flask run
# Visit: http://localhost:5000
```

**Option 2: Update Your ML Code**
```python
# In ESI_Engine.py or your code:
from adaptable_engine import get_config
config = get_config()
# Use config['environment']['device'] for device selection
# Use config['performance']['batch_size'] for batch size
# Use config['models']['gguf']['model_path'] for model path
```

**Option 3: Deploy to Production**
```bash
# Run same scripts on production machine
# Auto-detects and configures for that environment
# Done!
```

---

## 🤔 Questions?

| Question | Answer |
|----------|--------|
| **How long does deployment take?** | 5 minutes |
| **Do I need to configure anything manually?** | No, it's automatic |
| **Will it work on my Linux server?** | Yes, run `bash deploy-linux.sh` |
| **What if I don't have a GPU?** | Use CPU settings, same code works |
| **Can I use this on multiple machines?** | Yes, same code, auto-adapts to each |
| **Is it offline-only?** | Yes, medical-grade safety |
| **Can I modify configuration?** | Advanced: use environment variables |

---

**That's it! You're ready to deploy! 🚀**

For questions, see the full guides listed above.
