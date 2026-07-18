#!/usr/bin/env python3
"""
Cross-Platform Configuration Engine
Handles Windows/Linux compatibility, colorama, paths, and environment setup
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Optional, Tuple

# ============================================================================
# PLATFORM DETECTION & SYSTEM INFO
# ============================================================================

def get_platform_info() -> Dict[str, any]:
    """Detect platform and return comprehensive system info."""
    system = platform.system()
    is_windows = system == "Windows"
    is_linux = system == "Linux"
    is_macos = system == "Darwin"
    
    return {
        'system': system,
        'is_windows': is_windows,
        'is_linux': is_linux,
        'is_macos': is_macos,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'python_executable': sys.executable,
        'user_home': str(Path.home()),
        'cwd': os.getcwd(),
        'path_separator': os.pathsep,
        'line_separator': os.linesep,
    }

PLATFORM = get_platform_info()
IS_WINDOWS = PLATFORM['is_windows']
IS_LINUX = PLATFORM['is_linux']


# ============================================================================
# COLORAMA - CROSS-PLATFORM TERMINAL COLORS
# ============================================================================

class Colorama:
    """Unified colorama interface that works on all platforms."""
    
    # Color codes (ANSI for Linux/Mac, Windows API for Windows)
    RED = '\033[91m' if not IS_WINDOWS else ''
    GREEN = '\033[92m' if not IS_WINDOWS else ''
    YELLOW = '\033[93m' if not IS_WINDOWS else ''
    BLUE = '\033[94m' if not IS_WINDOWS else ''
    CYAN = '\033[96m' if not IS_WINDOWS else ''
    MAGENTA = '\033[95m' if not IS_WINDOWS else ''
    WHITE = '\033[97m' if not IS_WINDOWS else ''
    RESET = '\033[0m' if not IS_WINDOWS else ''
    BOLD = '\033[1m' if not IS_WINDOWS else ''
    
    @staticmethod
    def init(autoreset=True):
        """Initialize colorama with autoreset."""
        try:
            import colorama
            colorama.init(autoreset=autoreset, strip=IS_WINDOWS)
            return True
        except ImportError:
            print("[colorama] Not installed, using basic colors")
            return False
    
    @staticmethod
    def colored(text: str, color: str = 'white') -> str:
        """Color text with automatic fallback on Windows."""
        try:
            from colorama import Fore, Style
            color_map = {
                'red': Fore.RED,
                'green': Fore.GREEN,
                'yellow': Fore.YELLOW,
                'blue': Fore.BLUE,
                'cyan': Fore.CYAN,
                'magenta': Fore.MAGENTA,
                'white': Fore.WHITE,
            }
            return color_map.get(color, Fore.WHITE) + text + Style.RESET_ALL
        except ImportError:
            # Fallback: return text as-is
            return text
    
    @staticmethod
    def print_status(message: str, status: str = 'info'):
        """Print status message with appropriate color."""
        status_map = {
            'error': Colorama.RED + '❌',
            'warning': Colorama.YELLOW + '⚠️',
            'success': Colorama.GREEN + '✅',
            'info': Colorama.BLUE + 'ℹ️',
        }
        prefix = status_map.get(status, '•')
        print(f"{prefix} {message}{Colorama.RESET}")


# ============================================================================
# PATH HANDLING - CROSS-PLATFORM
# ============================================================================

class CrossPlatformPath:
    """Unified path handling for Windows and Linux."""
    
    @staticmethod
    def normalize(path: str) -> str:
        """Normalize path for current platform."""
        if path is None:
            return None
        
        # Convert to Path object for normalization
        p = Path(path)
        normalized = str(p)
        
        # On Windows, convert forward slashes to backslashes
        if IS_WINDOWS:
            normalized = normalized.replace('/', '\\')
        else:
            normalized = normalized.replace('\\', '/')
        
        return normalized
    
    @staticmethod
    def join(*parts: str) -> str:
        """Join path parts for current platform."""
        return str(Path(*parts))
    
    @staticmethod
    def to_posix(path: str) -> str:
        """Convert path to POSIX format (Linux/Mac style)."""
        return str(Path(path)).replace('\\', '/')
    
    @staticmethod
    def to_windows(path: str) -> str:
        """Convert path to Windows format."""
        return str(Path(path)).replace('/', '\\')
    
    @staticmethod
    def absolute(path: str) -> str:
        """Get absolute path."""
        return str(Path(path).absolute())
    
    @staticmethod
    def exists(path: str) -> bool:
        """Check if path exists."""
        return Path(path).exists()
    
    @staticmethod
    def mkdir(path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create directory."""
        Path(path).mkdir(parents=parents, exist_ok=exist_ok)


# ============================================================================
# ENVIRONMENT VARIABLES - CROSS-PLATFORM
# ============================================================================

class EnvironmentConfig:
    """Unified environment variable handling."""
    
    @staticmethod
    def get(key: str, default: str = None, required: bool = False) -> str:
        """Get environment variable with fallback."""
        value = os.environ.get(key, default)
        
        if required and value is None:
            raise ValueError(f"Required environment variable not set: {key}")
        
        return value
    
    @staticmethod
    def set(key: str, value: str) -> None:
        """Set environment variable."""
        os.environ[key] = str(value)
    
    @staticmethod
    def get_or_set(key: str, default: str) -> str:
        """Get environment variable or set it to default."""
        value = os.environ.get(key)
        if value is None:
            os.environ[key] = default
            return default
        return value
    
    @staticmethod
    def python_home() -> str:
        """Get Python home directory."""
        return str(Path(sys.executable).parent.parent)
    
    @staticmethod
    def site_packages() -> str:
        """Get site-packages directory."""
        import site
        return site.getsitepackages()[0] if site.getsitepackages() else None


# ============================================================================
# TERMINAL / CONSOLE HANDLING
# ============================================================================

class TerminalConfig:
    """Unified terminal configuration."""
    
    @staticmethod
    def enable_virtual_terminal():
        """Enable virtual terminal on Windows (ANSI support)."""
        if IS_WINDOWS:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                mode = ctypes.c_ulong()
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                # Enable ANSI escape sequence support (0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING)
                mode.value |= 0x0004
                kernel32.SetConsoleMode(handle, mode)
                return True
            except Exception as e:
                print(f"[terminal] Warning: Could not enable VT on Windows: {e}")
                return False
        return True  # Already enabled on Linux
    
    @staticmethod
    def set_console_title(title: str) -> None:
        """Set console window title."""
        if IS_WINDOWS:
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleTitleW(title)
            except:
                pass
        else:
            # Linux/Mac terminal escape sequence
            print(f'\033]0;{title}\007', end='', flush=True)
    
    @staticmethod
    def clear_screen() -> None:
        """Clear terminal screen."""
        cmd = 'cls' if IS_WINDOWS else 'clear'
        os.system(cmd)


# ============================================================================
# SHELL COMMAND EXECUTION - CROSS-PLATFORM
# ============================================================================

class ShellCommand:
    """Cross-platform shell command execution."""
    
    @staticmethod
    def execute(command: str, shell: bool = True, capture_output: bool = False) -> Tuple[int, str, str]:
        """Execute shell command and return (exit_code, stdout, stderr)."""
        import subprocess
        
        try:
            if capture_output:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                )
                return result.returncode, result.stdout, result.stderr
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=os.getcwd()
                )
                return result.returncode, '', ''
        except Exception as e:
            return -1, '', str(e)
    
    @staticmethod
    def which(program: str) -> Optional[str]:
        """Find program in PATH."""
        import shutil
        return shutil.which(program)
    
    @staticmethod
    def activate_venv(venv_path: str) -> str:
        """Get activation command for virtual environment."""
        venv_path = CrossPlatformPath.normalize(venv_path)
        
        if IS_WINDOWS:
            return f"{venv_path}\\Scripts\\activate.bat"
        else:
            return f"source {venv_path}/bin/activate"


# ============================================================================
# FILE SYSTEM OPERATIONS - CROSS-PLATFORM
# ============================================================================

class FileSystem:
    """Unified file system operations."""
    
    @staticmethod
    def read_file(path: str, encoding: str = 'utf-8') -> str:
        """Read file contents."""
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_file(path: str, content: str, encoding: str = 'utf-8') -> None:
        """Write file contents."""
        CrossPlatformPath.mkdir(os.path.dirname(path))
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def append_file(path: str, content: str, encoding: str = 'utf-8') -> None:
        """Append to file."""
        with open(path, 'a', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def list_files(path: str, pattern: str = '*') -> list:
        """List files matching pattern."""
        return [str(p) for p in Path(path).glob(pattern)]
    
    @staticmethod
    def delete_file(path: str) -> None:
        """Delete file."""
        Path(path).unlink()
    
    @staticmethod
    def file_size(path: str) -> int:
        """Get file size in bytes."""
        return Path(path).stat().st_size


# ============================================================================
# GGUF MODEL PATH RESOLUTION
# ============================================================================

class GGUFPathResolver:
    """Resolve GGUF model paths across platforms."""
    
    @staticmethod
    def find_gguf_model(model_name: str = 'medgemma-4b-it-Q4_K_M') -> Optional[str]:
        """Find GGUF model file on system."""
        # Common locations
        search_paths = [
            # Windows
            r'C:\Users\{user}\Videos\nursehelpdesk\backend\ML\models\medgemma-4b-local',
            r'C:\Users\{user}\Downloads\models',
            r'D:\models',
            # Linux
            '/home/{user}/Videos/nursehelpdesk/backend/ML/models/medgemma-4b-local',
            '/home/{user}/models',
            '/opt/models',
            # Relative to script
            './models',
            '../backend/ML/models/medgemma-4b-local',
        ]
        
        user = os.environ.get('USER') or os.environ.get('USERNAME')
        
        for base_path in search_paths:
            # Expand user in path
            if '{user}' in base_path:
                base_path = base_path.replace('{user}', user)
            
            base_path = CrossPlatformPath.normalize(base_path)
            
            # Look for GGUF file
            for pattern in ['*.gguf', f'*{model_name}*.gguf']:
                files = FileSystem.list_files(base_path, pattern) if CrossPlatformPath.exists(base_path) else []
                if files:
                    return files[0]
        
        return None
    
    @staticmethod
    def get_gguf_path(default: str = None) -> str:
        """Get GGUF model path with fallback."""
        # Check environment variable first
        env_path = os.environ.get('GGUF_MODEL_PATH')
        if env_path and CrossPlatformPath.exists(env_path):
            return env_path
        
        # Try to find automatically
        found_path = GGUFPathResolver.find_gguf_model()
        if found_path:
            return found_path
        
        # Use default
        return default


# ============================================================================
# PYTHON ENVIRONMENT SETUP
# ============================================================================

class PythonEnvironment:
    """Manage Python environment setup."""
    
    @staticmethod
    def setup_offline_mode():
        """Configure environment for offline model operation."""
        base_dir = Path(__file__).resolve().parent
        cache_dir = base_dir / "model_cache"
        
        # Create cache directory
        CrossPlatformPath.mkdir(str(cache_dir))
        
        # Set Hugging Face environment variables
        env_vars = {
            "TRANSFORMERS_CACHE": str(cache_dir),
            "HF_HOME": str(cache_dir),
            "SENTENCE_TRANSFORMERS_HOME": str(cache_dir),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        
        for key, value in env_vars.items():
            EnvironmentConfig.set(key, value)
        
        return cache_dir
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check for required Python packages."""
        dependencies = {
            'flask': False,
            'torch': False,
            'transformers': False,
            'llama_cpp': False,
            'pandas': False,
            'numpy': False,
        }
        
        for package in dependencies.keys():
            try:
                __import__(package)
                dependencies[package] = True
            except ImportError:
                pass
        
        return dependencies


# ============================================================================
# INITIALIZATION & BOOTSTRAP
# ============================================================================

def initialize_cross_platform():
    """Initialize cross-platform environment."""
    print("\n" + "="*80)
    print("[cross-platform] Initializing Multi-Platform Environment")
    print("="*80)
    
    # Print system info
    print(f"\n[platform] System: {PLATFORM['system']}")
    print(f"[platform] Python: {PLATFORM['python_version']}")
    print(f"[platform] Home: {PLATFORM['user_home']}")
    
    # Enable virtual terminal on Windows
    if IS_WINDOWS:
        if TerminalConfig.enable_virtual_terminal():
            print("[terminal] ✅ Virtual Terminal (ANSI) enabled on Windows")
        else:
            print("[terminal] ⚠️  Virtual Terminal disabled - colors may not work")
    
    # Initialize colorama
    if Colorama.init():
        print("[colorama] ✅ Colorama initialized")
    else:
        print("[colorama] ⚠️  Colorama not available - using basic colors")
    
    # Setup offline mode
    cache_dir = PythonEnvironment.setup_offline_mode()
    print(f"[offline] ✅ Offline mode configured: {cache_dir}")
    
    # Check dependencies
    deps = PythonEnvironment.check_dependencies()
    print(f"\n[dependencies] Installation Status:")
    for pkg, installed in deps.items():
        status = Colorama.colored("✅ installed", "green") if installed else Colorama.colored("❌ missing", "red")
        print(f"[dependencies]   {pkg}: {status}")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# EXPORT PUBLIC API
# ============================================================================

__all__ = [
    'PLATFORM',
    'IS_WINDOWS',
    'IS_LINUX',
    'Colorama',
    'CrossPlatformPath',
    'EnvironmentConfig',
    'TerminalConfig',
    'ShellCommand',
    'FileSystem',
    'GGUFPathResolver',
    'PythonEnvironment',
    'initialize_cross_platform',
]
