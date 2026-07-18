#!/usr/bin/env python3
"""
Interactive Configuration Setup for ESI_Engine
Detects local GGUF models and RAG systems, displays specs, and configures performance
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import psutil
import platform


class TriageEngineConfig:
    """Interactive configuration for medical triage engine."""
    
    def __init__(self):
        self.gguf_path = None
        self.rag_path = None
        self.specs = {}
        self.config = {}
    
    def prompt_for_directory(self, prompt_text: str, check_contents: str = None) -> Optional[Path]:
        """
        Prompt user for directory path with validation.
        
        Args:
            prompt_text: Message to display
            check_contents: If provided, check for files matching this pattern
        
        Returns:
            Path object or None if invalid
        """
        while True:
            user_input = input(f"\n{prompt_text}\n> ").strip()
            
            if user_input.lower() == 'skip':
                print("⏭️  Skipped")
                return None
            
            path = Path(user_input).expanduser()
            
            if not path.exists():
                print(f"❌ Path does not exist: {path}")
                print("   (Type 'skip' to skip this step)")
                continue
            
            if not path.is_dir():
                print(f"❌ Not a directory: {path}")
                print("   (Type 'skip' to skip this step)")
                continue
            
            if check_contents:
                # Check if directory contains expected files
                files = list(path.glob(check_contents))
                if not files:
                    print(f"⚠️  No files matching '{check_contents}' found in directory")
                    confirm = input("   Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
            
            print(f"✅ Directory accepted: {path}")
            return path
    
    def detect_gguf_specs(self, gguf_path: Path) -> Dict:
        """
        Detect GGUF model files and extract specifications.
        
        Returns:
            Dict with model info, file size, expected performance
        """
        specs = {
            'gguf_files': [],
            'total_size_gb': 0,
            'largest_model': None,
            'largest_size_mb': 0,
            'quantization': 'unknown',
            'estimated_vram_gb': 0,
            'recommended_batch_size': 32,
            'recommended_ctx_size': 1200,
            'estimated_tokens_per_sec': 0
        }
        
        print(f"\n🔍 Scanning GGUF directory: {gguf_path}")
        
        # Find all GGUF files
        gguf_files = list(gguf_path.glob('**/*.gguf'))
        
        if not gguf_files:
            print("   ❌ No GGUF files found")
            return specs
        
        print(f"   ✅ Found {len(gguf_files)} GGUF file(s)")
        
        for gguf_file in gguf_files:
            size_bytes = gguf_file.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_bytes / (1024 ** 3)
            
            specs['gguf_files'].append({
                'name': gguf_file.name,
                'path': str(gguf_file),
                'size_mb': round(size_mb, 2),
                'size_gb': round(size_gb, 2)
            })
            
            specs['total_size_gb'] += size_gb
            
            # Track largest model
            if size_mb > specs['largest_size_mb']:
                specs['largest_size_mb'] = size_mb
                specs['largest_model'] = gguf_file.name
                
                # Estimate specs based on model size
                # 4B model (Q4_K_M) ≈ 2.5-3GB
                # 7B model (Q4_K_M) ≈ 4.5-5GB
                # 13B model (Q4_K_M) ≈ 8-9GB
                if size_gb < 3.5:
                    specs['quantization'] = 'Q4_K_M (4B equivalent)'
                    specs['estimated_vram_gb'] = 2.5
                    specs['recommended_batch_size'] = 2000
                    specs['recommended_ctx_size'] = 1200
                    specs['estimated_tokens_per_sec'] = 25  # CPU: ~25 tokens/sec for 4B Q4_K_M
                elif size_gb < 6:
                    specs['quantization'] = 'Q4_K_M (7B equivalent)'
                    specs['estimated_vram_gb'] = 4.5
                    specs['recommended_batch_size'] = 1600
                    specs['recommended_ctx_size'] = 2048
                    specs['estimated_tokens_per_sec'] = 18  # CPU: ~18 tokens/sec for 7B Q4_K_M
                else:
                    specs['quantization'] = 'Q4_K_M (13B equivalent)'
                    specs['estimated_vram_gb'] = 8
                    specs['recommended_batch_size'] = 1024
                    specs['recommended_ctx_size'] = 4096
                    specs['estimated_tokens_per_sec'] = 10  # CPU: ~10 tokens/sec for 13B Q4_K_M
            
            print(f"   • {gguf_file.name}")
            print(f"     Size: {size_gb:.2f}GB ({size_mb:.0f}MB)")
        
        print(f"\n   📊 GGUF Summary:")
        print(f"   • Total models: {len(gguf_files)}")
        print(f"   • Total size: {specs['total_size_gb']:.2f}GB")
        print(f"   • Largest: {specs['largest_model']} ({specs['largest_size_mb']:.1f}MB)")
        print(f"   • Quantization: {specs['quantization']}")
        
        return specs
    
    def detect_rag_specs(self, rag_path: Path) -> Dict:
        """
        Detect RAG/Knowledge Graph system specifications.
        
        Returns:
            Dict with RAG info, FAISS index size, etc.
        """
        specs = {
            'system_type': 'unknown',
            'embedding_model': 'unknown',
            'index_files': [],
            'total_size_mb': 0,
            'case_count_estimated': 0,
            'embedding_dimension': 0,
            'retrieval_speed_estimate': 'unknown'
        }
        
        print(f"\n🔍 Scanning RAG directory: {rag_path}")
        
        # Check for FAISS index
        faiss_files = list(rag_path.glob('*.index'))
        if faiss_files:
            specs['system_type'] = 'FAISS Vector Index'
            specs['index_files'].extend(faiss_files)
            print(f"   ✅ Found {len(faiss_files)} FAISS index file(s)")
            
            for idx_file in faiss_files:
                size_mb = idx_file.stat().st_size / (1024 * 1024)
                specs['total_size_mb'] += size_mb
                print(f"   • {idx_file.name}: {size_mb:.1f}MB")
        
        # Check for embeddings
        emb_files = list(rag_path.glob('*embeddings*')) + list(rag_path.glob('*vectors*'))
        if emb_files:
            print(f"   ✅ Found {len(emb_files)} embedding file(s)")
            for emb_file in emb_files:
                size_mb = emb_file.stat().st_size / (1024 * 1024)
                specs['total_size_mb'] += size_mb
                print(f"   • {emb_file.name}: {size_mb:.1f}MB")
        
        # Check for case database
        case_files = list(rag_path.glob('*.jsonl')) + list(rag_path.glob('*.json'))
        if case_files:
            specs['system_type'] = 'RAG with Case Database'
            print(f"   ✅ Found {len(case_files)} case database file(s)")
            
            for case_file in case_files:
                size_mb = case_file.stat().st_size / (1024 * 1024)
                specs['total_size_mb'] += size_mb
                
                # Estimate case count from file size
                # Average case ≈ 1-2KB
                if case_file.suffix == '.jsonl':
                    estimated_cases = case_file.stat().st_size / 1500  # ~1.5KB per case
                    specs['case_count_estimated'] = max(specs['case_count_estimated'], int(estimated_cases))
                
                print(f"   • {case_file.name}: {size_mb:.1f}MB")
        
        # Check for models directory
        models_dir = rag_path / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('**/*'))
            if model_files:
                print(f"   ✅ Found embedding models directory ({len(model_files)} files)")
                # Check for PubMedBERT or other embedding models
                for f in model_files:
                    if 'pubmed' in f.name.lower():
                        specs['embedding_model'] = 'PubMedBERT'
                    elif 'bge' in f.name.lower():
                        specs['embedding_model'] = 'BGE (BGE-base or similar)'
                    elif 'minilm' in f.name.lower():
                        specs['embedding_model'] = 'All-MiniLM-L6-v2'
        
        # Estimate embedding dimension
        if specs['embedding_model'] == 'PubMedBERT':
            specs['embedding_dimension'] = 768
            specs['retrieval_speed_estimate'] = '100-200ms per query (GPU) / 500-1000ms (CPU)'
        elif 'BGE' in specs['embedding_model']:
            specs['embedding_dimension'] = 768
            specs['retrieval_speed_estimate'] = '100-150ms per query (GPU) / 400-800ms (CPU)'
        elif 'MiniLM' in specs['embedding_model']:
            specs['embedding_dimension'] = 384
            specs['retrieval_speed_estimate'] = '50-100ms per query'
        
        print(f"\n   📊 RAG Summary:")
        print(f"   • System type: {specs['system_type']}")
        print(f"   • Embedding model: {specs['embedding_model']}")
        print(f"   • Embedding dimension: {specs['embedding_dimension']}")
        print(f"   • Total size: {specs['total_size_mb']:.1f}MB")
        if specs['case_count_estimated']:
            print(f"   • Estimated cases: ~{specs['case_count_estimated']:,}")
        print(f"   • Retrieval speed: {specs['retrieval_speed_estimate']}")
        
        return specs
    
    def detect_system_specs(self) -> Dict:
        """
        Detect system hardware specifications.
        
        Returns:
            Dict with CPU, RAM, GPU info
        """
        specs = {
            'platform': platform.system(),
            'cpu_model': 'Unknown',
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'ram_total_gb': psutil.virtual_memory().total / (1024 ** 3),
            'ram_available_gb': psutil.virtual_memory().available / (1024 ** 3),
            'gpu_available': False,
            'gpu_type': 'None'
        }
        
        # Get CPU model
        try:
            import cpuinfo
            specs['cpu_model'] = cpuinfo.get_cpu_info().get('brand_raw', 'Unknown CPU')
        except:
            specs['cpu_model'] = platform.processor()
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                specs['gpu_available'] = True
                specs['gpu_type'] = torch.cuda.get_device_name(0)
                specs['gpu_vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except:
            pass
        
        return specs
    
    def calculate_performance_config(self, system_specs: Dict, gguf_specs: Dict, rag_specs: Dict) -> Dict:
        """
        Calculate optimal performance configuration based on detected specs.
        
        Returns:
            Dict with optimized parameters for inference
        """
        config = {
            'mode': 'hybrid',  # 'hybrid', 'llm_only', 'rag_only'
            'gguf_config': {
                'n_threads': max(4, system_specs['cpu_cores_physical']),
                'n_threads_batch': max(2, system_specs['cpu_cores_physical'] // 2),
                'n_batch': gguf_specs.get('recommended_batch_size', 2000),
                'n_ctx': gguf_specs.get('recommended_ctx_size', 1200),
                'use_mmap': True,
                'use_mlock': False,
                'mul_mat_q': True
            },
            'rag_config': {
                'enabled': rag_specs['system_type'] != 'unknown',
                'retriever': rag_specs['embedding_model'],
                'batch_retrieval_size': 10,
                'similarity_threshold': 0.70
            },
            'performance_hints': []
        }
        
        # Determine optimal mode
        has_gpu = system_specs['gpu_available']
        has_rag = rag_specs['system_type'] != 'unknown'
        
        if has_gpu and has_rag:
            config['mode'] = 'hybrid'
            config['performance_hints'].append("✅ GPU + RAG available: Use HYBRID mode for best accuracy")
        elif has_gpu:
            config['mode'] = 'llm_only'
            config['performance_hints'].append("✅ GPU available: Use LLM-only mode for speed")
        else:
            config['mode'] = 'llm_only'
            config['performance_hints'].append("⚠️  CPU-only: Using LLM-only mode (no RAG)")
        
        # Adjust threading based on cores
        if system_specs['cpu_cores_physical'] <= 4:
            config['gguf_config']['n_threads'] = 4
            config['gguf_config']['n_threads_batch'] = 2
            config['performance_hints'].append("⚠️  Low core count detected: Limited parallelism")
        elif system_specs['cpu_cores_physical'] >= 16:
            config['gguf_config']['n_threads'] = 12
            config['gguf_config']['n_threads_batch'] = 8
            config['performance_hints'].append("✅ High core count: Full parallelism enabled")
        
        # Adjust batch size based on available RAM
        if system_specs['ram_available_gb'] < 4:
            config['gguf_config']['n_batch'] = 1024
            config['performance_hints'].append("⚠️  Low available RAM: Reduced batch size")
        elif system_specs['ram_available_gb'] > 16:
            config['gguf_config']['n_batch'] = 4096
            config['performance_hints'].append("✅ Abundant RAM: Increased batch size for throughput")
        
        # Memory locking recommendation
        if system_specs['ram_available_gb'] > gguf_specs.get('estimated_vram_gb', 2.5) + 4:
            config['gguf_config']['use_mlock'] = True
            config['performance_hints'].append("💾 Enough RAM: Memory locking enabled for speed")
        
        # RAG retrieval optimization
        if config['rag_config']['enabled']:
            if rag_specs['case_count_estimated'] > 50000:
                config['rag_config']['batch_retrieval_size'] = 5
                config['performance_hints'].append(f"📚 Large case database ({rag_specs['case_count_estimated']:,} cases): Reduced batch retrieval")
        
        return config
    
    def display_full_config(self):
        """Display comprehensive configuration report."""
        print("\n" + "="*80)
        print("🏥 ESI TRIAGE ENGINE - FULL CONFIGURATION REPORT")
        print("="*80)
        
        print("\n📍 GGUF AI Model Configuration:")
        print("-" * 40)
        if self.specs.get('gguf'):
            gguf = self.specs['gguf']
            print(f"✅ Path: {self.gguf_path}")
            print(f"   Files: {len(gguf['gguf_files'])}")
            print(f"   Total size: {gguf['total_size_gb']:.2f}GB")
            print(f"   Primary model: {gguf['largest_model']}")
            print(f"   Quantization: {gguf['quantization']}")
            print(f"   Estimated VRAM: {gguf['estimated_vram_gb']}GB")
        else:
            print("❌ Not configured")
        
        print("\n📚 RAG/Knowledge Graph Configuration:")
        print("-" * 40)
        if self.specs.get('rag'):
            rag = self.specs['rag']
            print(f"✅ Path: {self.rag_path}")
            print(f"   System type: {rag['system_type']}")
            print(f"   Embedding model: {rag['embedding_model']}")
            print(f"   Total size: {rag['total_size_mb']:.1f}MB")
            if rag['case_count_estimated']:
                print(f"   Cases: ~{rag['case_count_estimated']:,}")
            print(f"   Retrieval speed: {rag['retrieval_speed_estimate']}")
        else:
            print("❌ Not configured")
        
        print("\n🖥️  System Specifications:")
        print("-" * 40)
        if self.specs.get('system'):
            sys_spec = self.specs['system']
            print(f"Platform: {sys_spec['platform']}")
            print(f"CPU: {sys_spec['cpu_model']}")
            print(f"   Cores: {sys_spec['cpu_cores_physical']} physical / {sys_spec['cpu_cores_logical']} logical")
            print(f"RAM: {sys_spec['ram_total_gb']:.1f}GB total / {sys_spec['ram_available_gb']:.1f}GB available")
            if sys_spec['gpu_available']:
                print(f"GPU: ✅ {sys_spec['gpu_type']} ({sys_spec.get('gpu_vram_gb', 0):.1f}GB VRAM)")
            else:
                print(f"GPU: ❌ None (CPU-only mode)")
        
        print("\n⚙️  Optimized Performance Configuration:")
        print("-" * 40)
        if self.config:
            print(f"Operating mode: {self.config['mode'].upper()}")
            print(f"\nGGUF Inference:")
            for key, val in self.config['gguf_config'].items():
                print(f"   {key}: {val}")
            print(f"\nRAG System:")
            for key, val in self.config['rag_config'].items():
                print(f"   {key}: {val}")
            print(f"\nPerformance Hints:")
            for hint in self.config['performance_hints']:
                print(f"   {hint}")
        
        print("\n" + "="*80)
    
    def save_config(self, output_file: str = 'triage_config.json'):
        """Save configuration to JSON file."""
        config_data = {
            'paths': {
                'gguf': str(self.gguf_path) if self.gguf_path else None,
                'rag': str(self.rag_path) if self.rag_path else None
            },
            'specs': self.specs,
            'config': self.config
        }
        
        output_path = Path(output_file)
        output_path.write_text(json.dumps(config_data, indent=2))
        print(f"\n💾 Configuration saved to: {output_path}")
        return output_path
    
    def run_interactive_setup(self) -> Tuple[Dict, Dict]:
        """
        Run complete interactive setup wizard.
        
        Returns:
            Tuple of (specs_dict, config_dict)
        """
        print("\n" + "="*80)
        print("🏥 ESI TRIAGE ENGINE - INTERACTIVE SETUP WIZARD")
        print("="*80)
        print("\nThis wizard will help you configure your medical triage engine.")
        print("Type 'skip' to skip any step.\n")
        
        # Step 1: GGUF Model
        print("\n[STEP 1/3] GGUF AI MODEL")
        print("-" * 40)
        print("Enter path to directory containing GGUF model files (*.gguf)")
        self.gguf_path = self.prompt_for_directory(
            "Enter GGUF directory path:",
            check_contents="*.gguf"
        )
        
        if self.gguf_path:
            self.specs['gguf'] = self.detect_gguf_specs(self.gguf_path)
        
        # Step 2: RAG/Knowledge Graph
        print("\n[STEP 2/3] RAG/KNOWLEDGE GRAPH SYSTEM")
        print("-" * 40)
        print("Enter path to directory containing RAG/knowledge graph files")
        print("(FAISS indexes, embeddings, or case databases)")
        self.rag_path = self.prompt_for_directory(
            "Enter RAG directory path:"
        )
        
        if self.rag_path:
            self.specs['rag'] = self.detect_rag_specs(self.rag_path)
        
        # Step 3: System Detection & Config
        print("\n[STEP 3/3] SYSTEM DETECTION & OPTIMIZATION")
        print("-" * 40)
        print("Detecting system hardware specifications...")
        self.specs['system'] = self.detect_system_specs()
        
        # Calculate optimized config
        print("\nCalculating optimized performance configuration...")
        self.config = self.calculate_performance_config(
            self.specs.get('system', {}),
            self.specs.get('gguf', {}),
            self.specs.get('rag', {})
        )
        
        # Display full report
        self.display_full_config()
        
        # Save configuration
        save_config = input("\n💾 Save configuration to file? (y/n): ").strip().lower()
        if save_config == 'y':
            self.save_config()
        
        return self.specs, self.config


# ============================================================================
# ENVIRONMENT VARIABLE EXPORT
# ============================================================================

def export_config_to_env(config: Dict, gguf_path: Optional[Path] = None, rag_path: Optional[Path] = None):
    """
    Export configuration to environment variables for use in ESI_Engine.
    
    Sets:
    - GGUF_MODEL_PATH: Path to GGUF model file
    - KNOWLEDGE_GRAPH_DIR: Path to RAG system
    - GGUF_N_THREADS: Thread count
    - GGUF_N_BATCH: Batch size
    - GGUF_N_CTX: Context window
    - DISABLE_RAG: Whether to disable RAG (0=enabled, 1=disabled)
    """
    env_exports = {}
    
    # Find largest GGUF file
    if gguf_path and gguf_path.exists():
        gguf_files = list(gguf_path.glob('**/*.gguf'))
        if gguf_files:
            # Sort by size, take largest
            largest = max(gguf_files, key=lambda f: f.stat().st_size)
            env_exports['GGUF_MODEL_PATH'] = str(largest)
    
    # Set RAG path
    if rag_path and rag_path.exists():
        env_exports['KNOWLEDGE_GRAPH_DIR'] = str(rag_path)
        env_exports['DISABLE_RAG'] = '0'  # Enable RAG
    else:
        env_exports['DISABLE_RAG'] = '1'  # Disable RAG if not available
    
    # Set GGUF parameters
    if config and config.get('gguf_config'):
        gguf_cfg = config['gguf_config']
        env_exports['GGUF_N_THREADS'] = str(gguf_cfg.get('n_threads', 6))
        env_exports['GGUF_N_BATCH'] = str(gguf_cfg.get('n_batch', 2000))
        env_exports['GGUF_N_CTX'] = str(gguf_cfg.get('n_ctx', 1200))
    
    return env_exports


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Run interactive setup
    setup_wizard = TriageEngineConfig()
    specs, config = setup_wizard.run_interactive_setup()
    
    # Export to environment variables
    env_vars = export_config_to_env(config, setup_wizard.gguf_path, setup_wizard.rag_path)
    
    print("\n" + "="*80)
    print("🔧 ENVIRONMENT VARIABLES TO SET:")
    print("="*80)
    for key, value in env_vars.items():
        print(f"export {key}='{value}'")
    
    print("\n💡 TIP: Copy the above commands to set environment variables before running ESI_Engine.py")
    print("\n✅ Setup complete! You can now run: python ESI_Engine.py")
