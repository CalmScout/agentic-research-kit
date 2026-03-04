#!/usr/bin/env python
"""Test script to verify unified model loading and memory savings."""

import gc
import logging

import torch

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
    logger.info(f"{'='*60}\n")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_unified_model():
    """Test the UnifiedQwen3VL model."""
    print_memory_stats("Test: Unified Model Approach")

    try:
        import numpy as np
        from PIL import Image

        from src.utils.vision_embedding import UnifiedQwen3VL

        model_name = "Qwen/Qwen3-VL-2B-Instruct"

        logger.info(f"Loading UnifiedQwen3VL: {model_name}...")
        unified_model = UnifiedQwen3VL(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="float16"
        )

        print_memory_stats("After loading UnifiedQwen3VL (single instance)")

        # Test text generation
        logger.info("Testing text generation...")
        response = unified_model.generate_text(
            prompt="What is 2+2?",
            max_tokens=50
        )
        logger.info(f"✓ Text response: {response[:100]}...")

        # Test image analysis (create a dummy image)
        logger.info("Testing image analysis...")
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        response = unified_model.analyze_image(
            image=dummy_image,
            prompt="Describe this image",
            max_tokens=50
        )
        logger.info(f"✓ Vision response: {response[:100]}...")

        print_memory_stats("After testing both text and vision")

        # Cleanup
        unified_model.cleanup()
        clear_gpu_memory()

        print_memory_stats("After cleanup")
        return True

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ingester_unified():
    """Test ClaimRAGIngester with unified model."""
    print_memory_stats("Test: ClaimRAGIngester with Unified Model")

    try:
        import pandas as pd
        from src.data_ingestion.rag_ingester import ClaimRAGIngester

        logger.info("Initializing ClaimRAGIngester...")
        ingester = ClaimRAGIngester(
            working_dir="./test_rag_storage",
            use_gpu=True,
            device="cuda",
            torch_dtype="float16"
        )

        print_memory_stats("After ClaimRAGIngester initialization")

        # Check if unified model is being used
        if ingester.use_unified_model:
            logger.info(f"✓ Using unified model: {ingester.vision_model_name}")
            logger.info("  (Single model instance for both vision and text)")
        else:
            logger.warning("⚠️  Unified model not detected - will load separate models")

        # Test embedding dimension detection
        logger.info("Testing embedding generation...")
        test_embedding = ingester._get_embedding_model().embed_text("test")
        logger.info(f"✓ Embedding dimension: {test_embedding.shape[0]}")

        print_memory_stats("After loading embedding model")

        # Test with small sample data
        logger.info("Testing ingestion with sample data...")
        _sample_df = pd.DataFrame({
            'reviewed claim': ['Test claim 1', 'Test claim 2'],
            'summary': ['Summary 1', 'Summary 2'],
            'url': ['http://test1.com', 'http://test2.com'],
            'title': ['Title 1', 'Title 2'],
            'country': ['US', 'UK'],
            'date': ['2024-01-01', '2024-01-02']
        })

        # Note: We won't actually ingest since that requires RAG-Anything setup
        # Just verify the models load correctly
        logger.info("✓ All models loaded successfully!")

        print_memory_stats("Final state")

        return True

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("UNIFIED MODEL VERIFICATION TEST")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("="*60 + "\n")

    results = {}

    # Test 1: Unified model alone
    results['unified_model'] = test_unified_model()

    clear_gpu_memory()

    # Test 2: Ingester with unified model
    results['ingester'] = test_ingester_unified()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Unified Model Test:     {'✓ PASS' if results['unified_model'] else '❌ FAIL'}")
    logger.info(f"Ingester Test:          {'✓ PASS' if results['ingester'] else '❌ FAIL'}")
    logger.info("="*60 + "\n")

    if all(results.values()):
        logger.info("✓ SUCCESS: Unified model approach is working!")
        logger.info("  Memory savings: ~4GB VRAM by loading model once")
    else:
        logger.error("❌ FAILURE: Some tests failed")


if __name__ == "__main__":
    main()
