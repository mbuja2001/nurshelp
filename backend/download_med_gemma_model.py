#!/usr/bin/env python3
"""
Download MedGemma-1.5-4B-IT model to local cache for offline use.
This script should be run ONCE before starting ESI_Engine.py
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configuration
MODEL_ID = "google/medgemma-1.5-4b-it"
CACHE_DIR = "/home/wtc/Documents/living_compute_labs/nursehelp/backend/ML/models"

print("=" * 80)
print("MedGemma-1.5-4B-IT Download Script")
print("=" * 80)
print()
print("This script will download the MedGemma-1.5-4B-IT model to local cache.")
print("The model with 8-bit quantization is ~4GB and will be stored in:", CACHE_DIR)
print()

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

print("[1] Downloading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    print("✅ Tokenizer downloaded successfully")
except Exception as e:
    print(f"❌ Failed to download tokenizer: {e}")
    exit(1)

print()
print("[2] Downloading model (8-bit quantized, ~4GB)...")
print("This may take several minutes...")
try:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ Model downloaded successfully (8-bit quantized)")
except Exception as e:
    print(f"❌ Failed to download model with 8-bit quantization: {e}")
    print("[2b] Retrying with fp16 fallback...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        print("✅ Model downloaded successfully (fp16 fallback)")
    except Exception as e2:
        print(f"❌ Failed to download model with fp16: {e2}")
        exit(1)

print()
print("=" * 80)
print("✅ SUCCESS: MedGemma-1.5-4B-IT is now cached locally!")
print("=" * 80)
print()
print("Model Details:")
print(f"  - Model ID: {MODEL_ID}")
print(f"  - Size: ~4GB (8-bit quantized)")
print(f"  - Cache location: {CACHE_DIR}")
print(f"  - Parameters: 4 billion")
print(f"  - Specialization: Medical/Clinical reasoning")
print(f"  - Chat format: <start_of_turn>user ... <end_of_turn>")
print()
print("You can now run ESI_Engine.py with:")
print("  FORCE_PURE_LLM=true USE_SUPERVISOR=true python3 ESI_Engine.py")
print()
