#!/usr/bin/env python3
"""
Download MedGemma GGUF model to the correct location.
This script handles the HuggingFace model download directly.
"""

import os
import sys
from pathlib import Path

def download_model():
    """Download GGUF model using huggingface-cli or requests."""
    
    # Target directory
    target_dir = Path(r"C:\Users\mbule\Videos\nursehelpdesk\nurshelp\backend\ML\ESI_Engine_Complete\data\models\medgemma-4b-local\models--unsloth--medgemma-4b-it-bnb-4bit")
    target_file = target_dir / "medgemma-4b-it-Q4_K_M-003.gguf"
    
    print(f"📁 Target directory: {target_dir}")
    print(f"📄 Target file: {target_file}")
    print()
    
    # Create directory if needed
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory created/verified: {target_dir}")
    print()
    
    # Try method 1: huggingface-cli
    print("=" * 80)
    print("METHOD 1: Using huggingface-cli (recommended)")
    print("=" * 80)
    print()
    print("Run this command to download the model:")
    print()
    cmd = f'huggingface-cli download unsloth/medgemma-4b-it-GGUF medgemma-4b-it-Q4_K_M.gguf --local-dir "{target_dir}"'
    print(f"  {cmd}")
    print()
    print("OR use alternative command:")
    print()
    cmd_alt = f'huggingface-cli download unsloth/medgemma-4b-it-GGUF medgemma-4b-it-Q4_K_M.gguf --local-dir-use-symlinks False --local-dir "{target_dir}"'
    print(f"  {cmd_alt}")
    print()
    print("=" * 80)
    print("METHOD 2: Manual download from HuggingFace")
    print("=" * 80)
    print()
    print("1. Visit: https://huggingface.co/unsloth/medgemma-4b-it-GGUF")
    print("2. Download: medgemma-4b-it-Q4_K_M.gguf (2.4GB)")
    print(f"3. Save to: {target_dir}")
    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()
    if target_file.exists():
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {target_file.name}")
        print(f"   Size: {size_mb:.1f} MB")
    else:
        print(f"⏳ Model file not yet downloaded")
        print(f"   Expected at: {target_file}")

if __name__ == "__main__":
    download_model()
