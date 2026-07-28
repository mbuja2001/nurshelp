#!/bin/bash
# Start ESI_Engine in LLM-ONLY mode (GGUF + deterministic rules, no RAG)
# This is ideal for CPU-only systems without GPU support

cd "$(dirname "$0")/backend/ML"

echo "🚀 Starting ESI Engine in LLM-ONLY mode (GGUF + deterministic rules)"
echo "   Environment: DISABLE_RAG=1"
echo "   This mode uses:"
echo "   - MedGemma 4B GGUF for clinical reasoning"
echo "   - Deterministic TEWS/ESI rules"
echo "   - No RAG embeddings (faster, lower CPU)"
echo ""

export DISABLE_RAG=1
python3 ESI_Engine.py
