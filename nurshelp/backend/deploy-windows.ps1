# Deployment script for Adaptable Triage Engine - Windows version
# Usage: powershell -ExecutionPolicy Bypass -File deploy-windows.ps1

$ErrorActionPreference = "Stop"

# ============================================================================
# CONFIGURATION
# ============================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Resolve-Path "$ScriptDir\..\..\..\"
$BackendDir = "$WorkspaceRoot\nurshelp\backend"
$MLDir = "$BackendDir\ML"
$VenvPath = "$MLDir\venv"

# Colors (using PSStyle for PowerShell 7+ / fallback for PS 5.1)
$ColorInfo = "Cyan"
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"

# ============================================================================
# FUNCTIONS
# ============================================================================

function Print-Step {
    param([string]$Message)
    Write-Host "[DEPLOY] $Message" -ForegroundColor $ColorInfo
}

function Print-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $ColorSuccess
}

function Print-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $ColorWarning
}

function Print-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $ColorError
}

function Test-CommandExists {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# ============================================================================
# DETECTION
# ============================================================================

Print-Step "Detecting system configuration..."

# OS Detection
$OS = "Windows"
$OSVersion = (Get-CimInstance Win32_OperatingSystem).Caption
Print-Success "OS: $OSVersion"

# Python Detection
if (-not (Test-CommandExists python)) {
    Print-Error "Python not found in PATH. Please install Python 3.9+"
    exit 1
}

$PythonVersion = python --version 2>&1 | Out-String
Print-Success "Python: $PythonVersion"

# Verify Python version
$PythonPath = (Get-Command python).Source
$PythonInfo = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
if ($PythonInfo -like "3.[0-8]") {
    Print-Error "Python 3.9+ required (found: $PythonInfo)"
    exit 1
}

# GPU Detection (NVIDIA)
$GPUAvailable = $false
if (Test-CommandExists nvidia-smi) {
    Print-Success "NVIDIA GPU detected"
    $GPUAvailable = $true
} else {
    Print-Warning "No NVIDIA GPU detected (will use CPU)"
    $GPUAvailable = $false
}

# System Information
$CPUInfo = Get-CimInstance Win32_Processor
$CPUCores = $CPUInfo.NumberOfLogicalProcessors
Print-Success "CPU Cores: $CPUCores"

$RAMInfo = Get-CimInstance Win32_OperatingSystem
$RAMBytes = $RAMInfo.TotalVisibleMemorySize * 1024
$RAMGB = [math]::Round($RAMBytes / 1GB, 1)
Print-Success "RAM: ${RAMGB}GB"

# ============================================================================
# WORKSPACE SETUP
# ============================================================================

Print-Step "Setting up workspace..."

if (-not (Test-Path $BackendDir)) {
    Print-Error "Backend directory not found: $BackendDir"
    exit 1
}

Print-Success "Workspace Root: $WorkspaceRoot"
Print-Success "Backend Dir: $BackendDir"

# ============================================================================
# VIRTUAL ENVIRONMENT
# ============================================================================

Print-Step "Setting up Python virtual environment..."

if (Test-Path $VenvPath) {
    Print-Warning "Virtual environment already exists at $VenvPath"
    $response = Read-Host "Delete and recreate? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Remove-Item -Path $VenvPath -Recurse -Force
        & python -m venv $VenvPath
        Print-Success "Virtual environment created"
    }
} else {
    & python -m venv $VenvPath
    Print-Success "Virtual environment created"
}

# Activate virtual environment
$ActivateScript = "$VenvPath\Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Print-Error "Virtual environment activation script not found"
    exit 1
}

& $ActivateScript

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel 2>&1 | Out-Null
Print-Success "pip upgraded"

# ============================================================================
# REQUIREMENTS INSTALLATION
# ============================================================================

Print-Step "Installing Python requirements..."

$RequirementsFile = "$MLDir\requirements.txt"

if (-not (Test-Path $RequirementsFile)) {
    Print-Error "requirements.txt not found at $RequirementsFile"
    exit 1
}

pip install -r $RequirementsFile 2>&1 | Out-Null
Print-Success "Requirements installed"

# Optional GPU support
if ($GPUAvailable) {
    Print-Step "Installing CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | Out-Null
    Print-Success "CUDA support installed"
} else {
    Print-Step "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio 2>&1 | Out-Null
    Print-Success "PyTorch installed (CPU mode)"
}

# ============================================================================
# CROSS-PLATFORM FILES
# ============================================================================

Print-Step "Verifying cross-platform support files..."

$CrossPlatformConfig = "$BackendDir\cross_platform_config.py"
$AdaptableEngine = "$BackendDir\adaptable_engine.py"

if (-not (Test-Path $CrossPlatformConfig)) {
    Print-Error "cross_platform_config.py not found"
    exit 1
}
Print-Success "cross_platform_config.py found"

if (-not (Test-Path $AdaptableEngine)) {
    Print-Error "adaptable_engine.py not found"
    exit 1
}
Print-Success "adaptable_engine.py found"

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

Print-Step "Creating required directories..."

$dirs = @(
    "$MLDir\models",
    "$BackendDir\model_cache",
    "$BackendDir\logs"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
    }
}

Print-Success "Directories created"

# ============================================================================
# CONFIGURATION TEST
# ============================================================================

Print-Step "Testing adaptable engine configuration..."

Push-Location $BackendDir

try {
    $output = python adaptable_engine.py 2>&1
    Print-Success "Engine configuration test passed"
    Write-Host ""
    Write-Host $output
} catch {
    Print-Error "Engine configuration test failed"
    Write-Host $_
    exit 1
} finally {
    Pop-Location
}

# ============================================================================
# MODEL SETUP
# ============================================================================

Print-Step "Checking for GGUF model..."

$ModelPath = Get-ChildItem -Path "$MLDir\models" -Name "*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $ModelPath) {
    Print-Warning "No GGUF model found in $MLDir\models\"
    Print-Warning "Expected: medgemma-4b-it-Q4_K_M.gguf"
    Print-Warning ""
    Print-Warning "To download the model, visit:"
    Print-Warning "  https://huggingface.co/search?q=medgemma-4b-it&type=model"
} else {
    $ModelFullPath = "$MLDir\models\$ModelPath"
    $ModelSize = (Get-Item $ModelFullPath).Length | ForEach-Object { if ($_ -gt 1GB) { "$([math]::Round($_ / 1GB, 1))GB" } else { "$([math]::Round($_ / 1MB, 1))MB" } }
    Print-Success "GGUF Model found: $ModelPath ($ModelSize)"
}

# ============================================================================
# DEPENDENCIES CHECK
# ============================================================================

Print-Step "Verifying all dependencies..."

Push-Location $BackendDir

$dependencies_check = @"
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
        if dep not in ['pathlib', 'platform', 'os', 'sys']:
            print(f"  ✗ {dep}")
            missing.append(dep)

if missing:
    print(f"\nMissing dependencies: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll dependencies verified!")
"@

$result = python -c $dependencies_check 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $result
    Print-Success "Dependency check passed"
} else {
    Write-Host $result
    Print-Error "Dependency check failed"
    Pop-Location
    exit 1
}

Pop-Location

# ============================================================================
# ENVIRONMENT SYNC
# ============================================================================

Print-Step "Creating environment lock file..."

$LockFile = "$MLDir\requirements.lock"
pip freeze | Out-File -Path $LockFile -Encoding UTF8
Print-Success "Environment lock file created: $LockFile"

# ============================================================================
# CREATE SHORTCUT
# ============================================================================

Print-Step "Creating convenience shortcuts..."

# PowerShell profile snippet for easy activation
$ProfileSnippet = @"
# NurseHelp Adaptable Engine - Quick Start
function Start-NurseHelp {
    Set-Location "$BackendDir"
    & "$VenvPath\Scripts\Activate.ps1"
    Write-Host "`nNurseHelp Backend Environment Activated!"
    Write-Host "Available commands:"
    Write-Host "  - python adaptable_engine.py (show configuration)"
    Write-Host "  - python -m flask run (start backend)"
    Write-Host "  - pip freeze (show all packages)"
}
"@

$ProfilePath = "$PROFILE"
if (Test-Path $ProfilePath) {
    Add-Content -Path $ProfilePath -Value $ProfileSnippet
    Print-Success "Added 'Start-NurseHelp' function to PowerShell profile"
    Print-Warning "Run 'Start-NurseHelp' to quickly activate the environment"
}

# ============================================================================
# SUMMARY
# ============================================================================

Write-Host ""
Write-Host "═════════════════════════════════════════════════════════" -ForegroundColor $ColorSuccess
Write-Host "✓ Adaptable Triage Engine Deployment Complete!" -ForegroundColor $ColorSuccess
Write-Host "═════════════════════════════════════════════════════════" -ForegroundColor $ColorSuccess
Write-Host ""
Write-Host "Configuration Summary:"
Write-Host "  OS: $OS"
Write-Host "  Python: $PythonInfo"
Write-Host "  GPU Available: $GPUAvailable"
Write-Host "  CPU Cores: $CPUCores"
Write-Host "  RAM: ${RAMGB}GB"
Write-Host "  Workspace: $WorkspaceRoot"
Write-Host "  Venv: $VenvPath"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor $ColorInfo
Write-Host "  1. Close and reopen PowerShell (to load profile)"
Write-Host "  2. Run: Start-NurseHelp"
Write-Host "  3. Test: python adaptable_engine.py"
Write-Host "  4. Run backend: python -m flask run"
Write-Host ""
Write-Host "Manual activation (if Start-NurseHelp not available):"
Write-Host "  & '$ActivateScript'"
Write-Host ""
