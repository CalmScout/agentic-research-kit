"""Test script to verify Qwen3-VL-Embedding-2B model loading."""

import torch
from transformers import AutoModel, Qwen3VLModel

print("Testing Qwen3-VL-Embedding-2B model loading...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("CUDA device: N/A")

model_name = "Qwen/Qwen3-VL-Embedding-2B"
print(f"\nLoading {model_name}...")

try:
    print("\n1. Loading with AutoModel...")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    print("✓ Model loaded successfully with AutoModel!")

except Exception as e:
    print(f"❌ Failed with AutoModel: {e}")

    try:
        print("\n2. Loading with Qwen3VLModel...")
        model = Qwen3VLModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("✓ Model loaded successfully with Qwen3VLModel!")

    except Exception as e:
        print(f"❌ Failed with Qwen3VLModel: {e}")

print("\nDone!")
