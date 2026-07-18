#!/usr/bin/env python3
"""
Adaptable Triage Engine
Dynamically configures for Windows/Linux, CPU/GPU, with/without RAG
"""

import os
import sys
from pathlib import Path

# Import cross-platform utilities
from cross_platform_config import (
    Colorama,
    CrossPlatformPath,
    EnvironmentConfig,
    PythonEnvironment,
    GGUFPathResolver,
    initialize_cross_platform,
    IS_WINDOWS,
    IS_LINUX,
    PLATFORM,
)

# ============================================================================
# ADAPTABLE ENGINE CONFIGURATION
# ============================================================================

class AdaptableTriageEngine:
    """
    Adaptable medical triage engine that configures itself based on:
    - Platform (Windows/Linux/Mac)
    - Hardware (GPU/CPU)
    - Available models (GGUF, RAG, etc.)
    - Environment (dev/prod, offline/online)
    """
    
    def __init__(self):
        self.platform = PLATFORM
        self.config = self._build_config()
        self.initialized = False
    
    def _build_config(self) -> dict:
        """Build comprehensive engine configuration."""
        config = {
            # Platform settings
            'platform': {
                'system': PLATFORM['system'],
                'is_windows': IS_WINDOWS,
                'is_linux': IS_LINUX,
                'path_separator': os.pathsep,
                'line_separator': os.linesep,
            },
            
            # Path configuration (platform-adaptive)
            'paths': self._resolve_paths(),
            
            # Hardware configuration (auto-detect)
            'hardware': self._detect_hardware(),
            
            # Model configuration (adaptive loading)
            'models': self._configure_models(),
            
            # Environment configuration
            'environment': self._configure_environment(),
            
            # Feature flags (based on availability)
            'features': self._configure_features(),
            
            # Performance tuning
            'performance': self._configure_performance(),
        }
        
        return config
    
    def _resolve_paths(self) -> dict:
        """Resolve paths for both Windows and Linux."""
        # Auto-detect workspace root
        script_dir = Path(__file__).parent
        workspace_root = script_dir.parent.parent  # Go up from backend/
        
        # Platform-specific model locations
        if IS_WINDOWS:
            model_paths = [
                f"C:\\Users\\{os.environ.get('USERNAME', 'user')}\\Videos\\nursehelpdesk\\backend\\ML\\models",
                "D:\\models",
                workspace_root / "backend" / "ML" / "models",
            ]
        else:  # Linux/Mac
            model_paths = [
                f"/home/{os.environ.get('USER', 'user')}/Videos/nursehelpdesk/backend/ML/models",
                "/opt/models",
                workspace_root / "backend" / "ML" / "models",
            ]
        
        return {
            'workspace_root': str(workspace_root),
            'backend_dir': str(workspace_root / 'backend'),
            'ml_dir': str(workspace_root / 'backend' / 'ML'),
            'model_paths': [CrossPlatformPath.normalize(str(p)) for p in model_paths],
            'cache_dir': str(workspace_root / 'backend' / 'model_cache'),
            'logs_dir': str(workspace_root / 'backend' / 'logs'),
            'temp_dir': CrossPlatformPath.normalize(os.environ.get('TEMP') or os.environ.get('TMP', '/tmp')),
        }
    
    def _detect_hardware(self) -> dict:
        """Auto-detect available hardware."""
        hardware = {
            'cpu_cores': 0,
            'physical_cores': 0,
            'total_ram_gb': 0,
            'available_ram_gb': 0,
            'gpu_available': False,
            'gpu_type': 'none',
            'gpu_vram_gb': 0,
            'cuda_available': False,
            'cuda_version': None,
        }
        
        try:
            import psutil
            hardware['cpu_cores'] = psutil.cpu_count(logical=True)
            hardware['physical_cores'] = psutil.cpu_count(logical=False)
            hardware['total_ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
            hardware['available_ram_gb'] = round(psutil.virtual_memory().available / (1024**3), 1)
        except ImportError:
            pass
        
        try:
            import torch
            hardware['cuda_available'] = torch.cuda.is_available()
            if hardware['cuda_available']:
                hardware['gpu_available'] = True
                hardware['gpu_type'] = 'cuda'
                hardware['gpu_vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                hardware['cuda_version'] = torch.version.cuda
        except ImportError:
            pass
        
        return hardware
    
    def _configure_models(self) -> dict:
        """Configure available models with platform-specific paths."""
        models = {
            'gguf': {
                'available': False,
                'model_name': 'medgemma-4b-it-Q4_K_M',
                'model_path': None,
                'model_size_gb': 2.5,
                'quantization': 'Q4_K_M',
                'context_size': 1200,
                'batch_size': 2000,
            },
            'rag': {
                'available': False,
                'retriever_type': 'pubmedbert',
                'index_path': None,
                'similarity_threshold': 0.70,
            },
            'transformers': {
                'available': False,
                'cache_location': self.config['paths']['cache_dir'],
            },
        }
        
        # Try to find GGUF model
        gguf_path = GGUFPathResolver.get_gguf_path()
        if gguf_path:
            models['gguf']['model_path'] = CrossPlatformPath.normalize(gguf_path)
            models['gguf']['available'] = True
        
        # Check for transformers
        try:
            import transformers
            models['transformers']['available'] = True
        except ImportError:
            pass
        
        return models
    
    def _configure_environment(self) -> dict:
        """Configure environment variables."""
        env = {
            'offline_mode': True,
            'debug': os.environ.get('DEBUG', '0') == '1',
            'backend_port': int(os.environ.get('BACKEND_PORT', 5000)),
            'device': 'cuda' if self.config['hardware']['gpu_available'] else 'cpu',
            'disable_rag': os.environ.get('DISABLE_RAG', '0') == '1',
            'force_pure_llm': os.environ.get('FORCE_PURE_LLM', '0') == '1',
            'environment': os.environ.get('ENVIRONMENT', 'development'),
        }
        
        return env
    
    def _configure_features(self) -> dict:
        """Configure feature flags based on availability."""
        features = {
            'gguf_inference': self.config['models']['gguf']['available'],
            'rag_retrieval': self.config['models']['rag']['available'],
            'gpu_acceleration': self.config['hardware']['gpu_available'],
            'colorama_support': True,
            'offline_operation': True,
        }
        
        return features
    
    def _configure_performance(self) -> dict:
        """Configure performance tuning based on hardware."""
        hw = self.config['hardware']
        
        # CPU-optimized defaults
        performance = {
            'max_threads': min(8, hw['physical_cores']),
            'inference_threads': max(4, hw['physical_cores'] - 2),
            'batch_size': 2000,
            'context_size': 1200,
            'use_memory_mapping': True,
            'memory_budget_pct': 90,  # Use up to 90% of available RAM
        }
        
        # GPU-specific tuning
        if hw['gpu_available']:
            performance['gpu_batch_size'] = 32
            performance['gpu_memory_fraction'] = 0.8
            performance['use_mixed_precision'] = True
        
        return performance
    
    def initialize(self) -> bool:
        """Initialize the engine with detected configuration."""
        Colorama.print_status("Initializing Adaptable Triage Engine", "info")
        
        print(f"\n[engine] Platform Configuration:")
        print(f"[engine]   System: {self.config['platform']['system']}")
        print(f"[engine]   Python: {PLATFORM['python_version']}")
        
        print(f"\n[engine] Hardware Detection:")
        print(f"[engine]   CPU Cores: {self.config['hardware']['physical_cores']} physical")
        print(f"[engine]   RAM: {self.config['hardware']['available_ram_gb']} GB available")
        print(f"[engine]   GPU: {'Yes' if self.config['hardware']['gpu_available'] else 'No'}")
        
        print(f"\n[engine] Model Configuration:")
        print(f"[engine]   GGUF: {'✅ Available' if self.config['models']['gguf']['available'] else '❌ Not found'}")
        if self.config['models']['gguf']['model_path']:
            print(f"[engine]      Path: {self.config['models']['gguf']['model_path']}")
        print(f"[engine]   RAG: {'✅ Available' if self.config['models']['rag']['available'] else '❌ Not available'}")
        
        print(f"\n[engine] Feature Flags:")
        for feature, enabled in self.config['features'].items():
            status = Colorama.colored("enabled", "green") if enabled else Colorama.colored("disabled", "red")
            print(f"[engine]   {feature}: {status}")
        
        self.initialized = True
        Colorama.print_status("Engine initialized successfully", "success")
        return True
    
    def get_config(self) -> dict:
        """Get full configuration as dictionary."""
        return self.config
    
    def get_summary(self) -> str:
        """Get human-readable configuration summary."""
        summary = f"""
╔═══════════════════════════════════════════════════════════════╗
║         Adaptable Triage Engine Configuration Summary          ║
╠═══════════════════════════════════════════════════════════════╣
║ Platform: {self.config['platform']['system']:<50}║
║ Device: {self.config['environment']['device']:<52}║
║ GPU: {'Yes (' + str(self.config['hardware']['gpu_vram_gb']) + 'GB)' if self.config['hardware']['gpu_available'] else 'No':<50}║
║ RAM: {str(self.config['hardware']['available_ram_gb']) + ' GB available':<50}║
║ Offline Mode: {'Enabled' if self.config['environment']['offline_mode'] else 'Disabled':<40}║
╠═══════════════════════════════════════════════════════════════╣
║ Models Available:                                             ║
║   • GGUF: {'✅' if self.config['models']['gguf']['available'] else '❌':<46}║
║   • RAG: {'✅' if self.config['models']['rag']['available'] else '❌':<47}║
║ Features Enabled:                                             ║
║   • GPU Acceleration: {'✅' if self.config['features']['gpu_acceleration'] else '❌':<38}║
║   • GGUF Inference: {'✅' if self.config['features']['gguf_inference'] else '❌':<40}║
║   • RAG Retrieval: {'✅' if self.config['features']['rag_retrieval'] else '❌':<41}║
╚═══════════════════════════════════════════════════════════════╝
"""
        return summary


# ============================================================================
# GLOBAL ENGINE INSTANCE
# ============================================================================

# Initialize at module load
initialize_cross_platform()
ENGINE = AdaptableTriageEngine()
ENGINE.initialize()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_engine() -> AdaptableTriageEngine:
    """Get global engine instance."""
    return ENGINE

def get_config() -> dict:
    """Get engine configuration."""
    return ENGINE.get_config()

def print_config_summary():
    """Print configuration summary."""
    print(ENGINE.get_summary())


if __name__ == '__main__':
    # Print full configuration when run as script
    import json
    print_config_summary()
    print("\n[engine] Full Configuration (JSON):")
    print(json.dumps(ENGINE.get_config(), indent=2, default=str))
