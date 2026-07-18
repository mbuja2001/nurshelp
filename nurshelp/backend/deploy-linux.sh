#!/usr/bin/env bash
# Deployment script for Adaptable Triage Engine - Linux version
# Usage: bash deploy-linux.sh

set -e  # Exit on any error

# Colors (ANSI - works on Linux/Mac)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# ============================================================================
# FUNCTIONS
# ============================================================================

print_step() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================================
# DETECTION
# ============================================================================

print_step "Detecting system configuration..."

# OS Detection
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    DISTRO=$(lsb_release -si 2>/dev/null || echo "unknown")
    print_success "OS: Linux ($DISTRO)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    print_success "OS: macOS"
else
    print_error "Unsupported OS: $OSTYPE"
    exit 1
fi

# Python Detection
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.9 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
print_success "Python: $PYTHON_VERSION"

# Check Python version (3.9+)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    print_error "Python 3.9 or later required (found: $PYTHON_VERSION)"
    exit 1
fi

# GPU Detection
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    GPU_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected (will use CPU)"
    GPU_AVAILABLE=false
fi

# CPU Information
CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
print_success "CPU Cores: $CORES"

# RAM Information
RAM_GB=$(($(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || vm_stat 2>/dev/null | grep "Pages free" | awk '{print $3}' | tr -d '.') / 1024 / 1024))
print_success "RAM: ${RAM_GB}GB available"

# ============================================================================
# WORKSPACE SETUP
# ============================================================================

print_step "Setting up workspace..."

# Auto-detect workspace root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
BACKEND_DIR="$WORKSPACE_ROOT/nurshelp/backend"
ML_DIR="$BACKEND_DIR/ML"

print_success "Workspace Root: $WORKSPACE_ROOT"
print_success "Backend Dir: $BACKEND_DIR"

# ============================================================================
# VIRTUAL ENVIRONMENT
# ============================================================================

print_step "Setting up Python virtual environment..."

VENV_PATH="$ML_DIR/venv"

if [ -d "$VENV_PATH" ]; then
    print_warning "Virtual environment already exists at $VENV_PATH"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        python3 -m venv "$VENV_PATH"
        print_success "Virtual environment created"
    fi
else
    python3 -m venv "$VENV_PATH"
    print_success "Virtual environment created"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "pip upgraded"

# ============================================================================
# REQUIREMENTS INSTALLATION
# ============================================================================

print_step "Installing Python requirements..."

REQUIREMENTS_FILE="$ML_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_error "requirements.txt not found at $REQUIREMENTS_FILE"
    exit 1
fi

pip install -r "$REQUIREMENTS_FILE" > /dev/null 2>&1
print_success "Requirements installed"

# Optional GPU support
if [ "$GPU_AVAILABLE" = true ]; then
    print_step "Installing CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
    print_success "CUDA support installed"
fi

# ============================================================================
# CROSS-PLATFORM FILES
# ============================================================================

print_step "Verifying cross-platform support files..."

CROSS_PLATFORM_CONFIG="$BACKEND_DIR/cross_platform_config.py"
ADAPTABLE_ENGINE="$BACKEND_DIR/adaptable_engine.py"

if [ ! -f "$CROSS_PLATFORM_CONFIG" ]; then
    print_error "cross_platform_config.py not found"
    exit 1
fi
print_success "cross_platform_config.py found"

if [ ! -f "$ADAPTABLE_ENGINE" ]; then
    print_error "adaptable_engine.py not found"
    exit 1
fi
print_success "adaptable_engine.py found"

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

print_step "Creating required directories..."

mkdir -p "$ML_DIR/models"
mkdir -p "$BACKEND_DIR/model_cache"
mkdir -p "$BACKEND_DIR/logs"

print_success "Directories created"

# ============================================================================
# CONFIGURATION TEST
# ============================================================================

print_step "Testing adaptable engine configuration..."

cd "$BACKEND_DIR"
python adaptable_engine.py > /tmp/engine_config.log 2>&1

if [ $? -eq 0 ]; then
    print_success "Engine configuration test passed"
    echo ""
    cat /tmp/engine_config.log
else
    print_error "Engine configuration test failed"
    cat /tmp/engine_config.log
    exit 1
fi

# ============================================================================
# MODEL SETUP
# ============================================================================

print_step "Checking for GGUF model..."

MODEL_PATH=$(find "$ML_DIR/models" -name "*.gguf" -type f 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
    print_warning "No GGUF model found in $ML_DIR/models/"
    print_warning "Expected: medgemma-4b-it-Q4_K_M.gguf"
    print_warning ""
    print_warning "To download the model, run:"
    print_warning "  wget -O $ML_DIR/models/medgemma-4b-it-Q4_K_M.gguf \\"
    print_warning "    https://huggingface.co/.../medgemma-4b-it-Q4_K_M.gguf"
else
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    print_success "GGUF Model found: $MODEL_PATH ($MODEL_SIZE)"
fi

# ============================================================================
# DEPENDENCIES CHECK
# ============================================================================

print_step "Verifying all dependencies..."

cd "$BACKEND_DIR"

python << 'EOF'
import sys

dependencies = {
    'pathlib': 'Standard Library',
    'platform': 'Standard Library',
    'os': 'Standard Library',
    'sys': 'Standard Library',
    'psutil': 'System utilities',
    'colorama': 'Cross-platform colors',
    'flask': 'Web framework',
    'click': 'CLI framework',
    'werkzeug': 'WSGI utilities',
    'python_dotenv': 'Environment config',
}

missing = []
for dep, description in dependencies.items():
    try:
        __import__(dep)
        print(f"  ✓ {dep}")
    except ImportError:
        if dep not in ['pathlib', 'platform', 'os', 'sys']:  # Skip stdlib
            print(f"  ✗ {dep}")
            missing.append(dep)

if missing:
    print(f"\nMissing dependencies: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll dependencies verified!")
EOF

if [ $? -eq 0 ]; then
    print_success "Dependency check passed"
else
    print_error "Dependency check failed"
    exit 1
fi

# ============================================================================
# ENVIRONMENT SYNC
# ============================================================================

print_step "Creating environment lock file..."

LOCK_FILE="$ML_DIR/requirements.lock"
pip freeze > "$LOCK_FILE"
print_success "Environment lock file created: $LOCK_FILE"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}═════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Adaptable Triage Engine Deployment Complete!${NC}"
echo -e "${GREEN}═════════════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration Summary:"
echo "  OS: $OS ($DISTRO)"
echo "  Python: $PYTHON_VERSION"
echo "  GPU Available: $GPU_AVAILABLE"
echo "  CPU Cores: $CORES"
echo "  RAM: ${RAM_GB}GB"
echo "  Workspace: $WORKSPACE_ROOT"
echo "  Venv: $VENV_PATH"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source $VENV_PATH/bin/activate"
echo "  2. Download GGUF model to: $ML_DIR/models/"
echo "  3. Test engine: cd $BACKEND_DIR && python adaptable_engine.py"
echo "  4. Run backend: cd $BACKEND_DIR && python -m flask run"
echo ""
echo "To run on next login:"
echo "  source $VENV_PATH/bin/activate"
echo "  cd $BACKEND_DIR"
echo ""
