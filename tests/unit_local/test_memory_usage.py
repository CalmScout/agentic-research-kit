#!/usr/bin/env python
"""Memory usage diagnostic test for Qwen3 models on 12GB GPU."""

import torch
import gc
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"available": 0, "used": 0, "total": 0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - allocated

    return {
        "total": total,
        "used": allocated,
        "free": free,
        "reserved": reserved
    }


def print_memory_stats(stage: str):
    """Print GPU memory statistics for a given stage."""
    stats = get_gpu_memory()
    logger.info(f"\n{'='*60}")
    logger.info(f"{stage}")
    logger.info(f"{'='*60}")
    logger.info(f"Total GPU Memory: {stats['total']:.2f} GB")
    logger.info(f"Used Memory:      {stats['used']:.2f} GB")
    logger.info(f"Free Memory:      {stats['free']:.2f} GB")
    logger.info(f"Reserved Memory:  {stats['reserved']:.2f} GB")
    logger.info(f"{'='*60}\n")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_baseline():
    """Test 1: Check baseline GPU memory."""
    print_memory_stats("Test 1: Baseline GPU Memory (before any models)")


def test_single_vl_model():
    """Test 2: Load Qwen3-VL-2B-Instruct ONCE."""
    print_memory_stats("Test 2: Before loading Qwen3-VL-2B-Instruct")

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        logger.info(f"Loading {model_name}...")

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print_memory_stats(f"Test 2: After loading {model_name} (ONCE)")

        # Cleanup
        del model
        del processor
        clear_gpu_memory()

        print_memory_stats("Test 2: After cleanup (should return to baseline)")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        print_memory_stats("Test 2: After failed load")
        return False


def test_embedding_model():
    """Test 3: Load Qwen3-VL-Embedding-2B."""
    print_memory_stats("Test 3: Before loading Qwen3-VL-Embedding-2B")

    try:
        from transformers import AutoModel, AutoTokenizer

        model_name = "Qwen/Qwen3-VL-Embedding-2B"
        logger.info(f"Loading {model_name}...")

        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print_memory_stats(f"Test 3: After loading {model_name}")

        # Test embedding generation
        logger.info("Testing embedding generation...")
        inputs = tokenizer("test text", return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

        logger.info(f"✓ Generated test embedding with shape: {embedding.shape}")
        print_memory_stats("Test 3: After embedding generation")

        # Cleanup
        del model
        del tokenizer
        clear_gpu_memory()

        print_memory_stats("Test 3: After cleanup")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        print_memory_stats("Test 3: After failed load")
        return False


def test_duplicate_loading():
    """Test 4: Test CURRENT pattern (duplicate loading)."""
    print_memory_stats("Test 4: Testing CURRENT pattern (loads model TWICE)")

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_name = "Qwen/Qwen3-VL-2B-Instruct"

        # Load first instance (text LLM)
        logger.info(f"Loading FIRST instance of {model_name} (as text LLM)...")
        model1 = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print_memory_stats("Test 4: After loading FIRST instance")

        # Load second instance (vision model) - THIS IS THE PROBLEM!
        logger.info(f"Loading SECOND instance of {model_name} (as vision model)...")
        model2 = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print_memory_stats("Test 4: After loading SECOND instance (DUPLICATE!)")
        logger.warning("⚠️  This is the MEMORY WASTE - same model loaded twice!")

        # Cleanup
        del model1
        del model2
        clear_gpu_memory()

        print_memory_stats("Test 4: After cleanup")
        return True

    except Exception as e:
        logger.error(f"❌ Failed duplicate loading test: {e}")
        print_memory_stats("Test 4: After failed load")
        return False


def test_unified_pattern():
    """Test 5: Test UNIFIED pattern (single model instance)."""
    print_memory_stats("Test 5: Testing UNIFIED pattern (loads model ONCE)")

    try:
        from src.utils.vision_embedding import UnifiedQwen3VL

        model_name = "Qwen/Qwen3-VL-2B-Instruct"

        logger.info(f"Loading UNIFIED {model_name} (single instance for both vision + text)...")
        unified_model = UnifiedQwen3VL(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="float16"
        )

        print_memory_stats("Test 5: After loading UNIFIED model (single instance)")
        logger.info("✓ This is the OPTIMIZED approach - one model for both tasks!")

        # Cleanup
        unified_model.cleanup()
        clear_gpu_memory()

        print_memory_stats("Test 5: After cleanup")
        return True

    except Exception as e:
        logger.error(f"❌ Failed unified pattern test: {e}")
        print_memory_stats("Test 5: After failed load")
        return False


def test_combined_models():
    """Test 6: Test BOTH models together (VL + Embedding)."""
    print_memory_stats("Test 6: Loading BOTH models (VL + Embedding)")

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModel, AutoTokenizer

        vl_model_name = "Qwen/Qwen3-VL-2B-Instruct"
        embed_model_name = "Qwen/Qwen3-VL-Embedding-2B"

        # Load VL model
        logger.info(f"Loading {vl_model_name}...")
        vl_model = AutoModelForImageTextToText.from_pretrained(
            vl_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print_memory_stats(f"Test 6: After loading {vl_model_name}")

        # Load Embedding model
        logger.info(f"Loading {embed_model_name}...")
        embed_model = AutoModel.from_pretrained(
            embed_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print_memory_stats(f"Test 6: After loading {embed_model_name}")
        logger.info("✓ Both models loaded successfully!")

        # Check if we're under 12GB
        stats = get_gpu_memory()
        if stats['used'] < 12.0:
            logger.info(f"✓ Memory usage ({stats['used']:.2f} GB) is within 12GB limit!")
        else:
            logger.error(f"❌ Memory usage ({stats['used']:.2f} GB) EXCEEDS 12GB limit!")

        # Cleanup
        del vl_model
        del embed_model
        clear_gpu_memory()

        print_memory_stats("Test 6: After cleanup")
        return stats['used'] < 12.0

    except Exception as e:
        logger.error(f"❌ Failed combined models test: {e}")
        print_memory_stats("Test 6: After failed load")
        return False


def main():
    """Run all memory tests."""
    logger.info("\n" + "="*60)
    logger.info("QWEN3 MODEL MEMORY USAGE DIAGNOSTIC TEST")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("="*60 + "\n")

    # Run tests
    results = {}

    test_baseline()
    results['single_vl'] = test_single_vl_model()
    results['embedding'] = test_embedding_model()
    results['duplicate'] = test_duplicate_loading()
    results['unified'] = test_unified_pattern()
    results['combined'] = test_combined_models()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Single VL Model:        {'✓ PASS' if results['single_vl'] else '❌ FAIL'}")
    logger.info(f"Embedding Model:        {'✓ PASS' if results['embedding'] else '❌ FAIL'}")
    logger.info(f"Duplicate Loading:      {'✓ PASS' if results['duplicate'] else '❌ FAIL'}")
    logger.info(f"Unified Pattern:        {'✓ PASS' if results['unified'] else '❌ FAIL'}")
    logger.info(f"Combined Models:        {'✓ PASS' if results['combined'] else '❌ FAIL'}")
    logger.info("="*60 + "\n")

    # Recommendations
    if results['combined']:
        logger.info("✓ SUCCESS: Both models can fit in 12GB GPU!")
        logger.info("  The pipeline should work with the current configuration.")
    else:
        logger.error("❌ FAILURE: Models exceed 12GB GPU limit!")
        logger.error("  RECOMMENDATION: Use UnifiedQwen3VL to eliminate duplicate loading.")

    logger.info("")


if __name__ == "__main__":
    main()
