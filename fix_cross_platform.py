#!/usr/bin/env python3
"""
Automated Cross-Platform Compatibility Fixer
Applies all cross-platform fixes to the NurseHelp backend
"""

import os
import sys
from pathlib import Path
import re

def fix_file(filepath, replacements):
    """Apply regex replacements to a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for pattern, replacement, description in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            if content != original_content:
                print(f"✅ {filepath}: {description}")
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"⏭️  {filepath}: No changes needed")
            return False
    except Exception as e:
        print(f"❌ {filepath}: {e}")
        return False

def main():
    backend_dir = Path(__file__).resolve().parent / "nurshelp" / "backend" / "ML"
    
    if not backend_dir.exists():
        print(f"❌ Backend directory not found: {backend_dir}")
        sys.exit(1)
    
    print(f"🔧 Cross-Platform Compatibility Fixer")
    print(f"📁 Target: {backend_dir}")
    print("=" * 80)
    
    # 1. Fix download_medgemma_model.py
    replacements_medgemma = [
        (
            r'cache_dir = "/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/models/medgemma-4b-local"',
            'cache_dir = str(Path(__file__).resolve().parent / "models" / "medgemma-4b-local")',
            "Fixed hardcoded path in download_medgemma_model.py"
        )
    ]
    
    fix_file(backend_dir / "download_medgemma_model.py", replacements_medgemma)
    
    # 2. Fix download_llama_model.py
    replacements_llama = [
        (
            r'cache_dir = "/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/models/tinyllama-1\.1b-local"',
            'cache_dir = str(Path(__file__).resolve().parent / "models" / "tinyllama-1.1b-local")',
            "Fixed hardcoded path in download_llama_model.py"
        )
    ]
    
    fix_file(backend_dir / "download_llama_model.py", replacements_llama)
    
    # 3. Fix all MODELS_DIR patterns
    models_dir_files = [
        "download_biogpt_model.py",
        "download_biomistral_2bit_model.py",
        "download_biomistral_4bit_model.py",
        "download_med_gemma_model.py",
        "download_medqwen_2bit_model.py",
        "download_meditron_model.py",
        "download_mistral_4bit_model.py",
        "download_mistral_model.py",
        "download_qwen_fp32_model.py",
        "download_tinyllama_2bit_model.py",
    ]
    
    for filename in models_dir_files:
        filepath = backend_dir / filename
        if filepath.exists():
            replacements = [
                (
                    r'MODELS_DIR = "/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/models"',
                    'MODELS_DIR = str(Path(__file__).resolve().parent / "models")',
                    f"Fixed MODELS_DIR in {filename}"
                ),
                (
                    r'MODELS_DIR = "/home/wtc/Videos/nursehelpdesk/backend/ML/models"',
                    'MODELS_DIR = str(Path(__file__).resolve().parent / "models")',
                    f"Fixed alternate MODELS_DIR path in {filename}"
                )
            ]
            fix_file(filepath, replacements)
    
    # 4. Fix download_clinical_datasets.py
    replacements_datasets = [
        (
            r'DATA_DIR = Path\("/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/rag_data"\)',
            'DATA_DIR = Path(__file__).resolve().parent / "rag_data"',
            "Fixed DATA_DIR path in download_clinical_datasets.py"
        )
    ]
    
    fix_file(backend_dir / "download_clinical_datasets.py", replacements_datasets)
    
    # 5. Fix inspect_pmc_parquet.py
    replacements_pmc = [
        (
            r"Path\('/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/pmc_cases_unique\.parquet'\)",
            'Path(__file__).resolve().parent / "pmc_cases_unique.parquet"',
            "Fixed pmc_cases_unique.parquet path"
        ),
        (
            r"Path\('/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/retrieval_augmented_generation/pmc_cases_unique\.parquet'\)",
            'Path(__file__).resolve().parent / "retrieval_augmented_generation" / "pmc_cases_unique.parquet"',
            "Fixed alternative pmc_cases_unique.parquet path"
        )
    ]
    
    fix_file(backend_dir / "inspect_pmc_parquet.py", replacements_pmc)
    
    # 6. Fix inspect_faiss_data.py
    replacements_faiss = [
        (
            r'print\("  cd /home/wtc/Documents/living_compute_labs/nursehelp/backend/ML"\)',
            'print(f"  cd {Path(__file__).resolve().parent}")',
            "Fixed dynamic cd path in inspect_faiss_data.py"
        )
    ]
    
    fix_file(backend_dir / "inspect_faiss_data.py", replacements_faiss)
    
    print("\n" + "=" * 80)
    print("✅ Cross-platform fixes applied!")
    print("\n📋 Next steps:")
    print("1. Create .env file with: GGUF_MODEL_PATH=<actual_path>")
    print("2. Run: python ESI_Engine.py")
    print("3. Check startup logs for ✅ GGUF from ...")
    print("\n💡 For environment setup, see CROSS_PLATFORM_COMPATIBILITY_FIXES.md")

if __name__ == "__main__":
    main()
