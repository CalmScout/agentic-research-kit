"""
Batch Ingestion Example

This example shows how to ingest a large dataset in batches,
with progress tracking and error handling.

Usage:
    uv run python examples/batch_ingest.py --dataset path/to/your/data.csv --max-items 100
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from src.data_ingestion.universal_pipeline import run_universal_pipeline


async def main():
    """Ingest dataset in batches with progress tracking."""

    parser = argparse.ArgumentParser(description="Batch ingest data into the ARK RAG system")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum number of items to ingest (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for ingestion (default: 10)"
    )

    args = parser.parse_args()

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return

    print("=" * 80)
    print("Agentic Research Kit (ARK) - Batch Ingestion")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Max Items: {args.max_items}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 80)
    print()

    try:
        # Load dataset
        print(f"📂 Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded {len(df)} rows from dataset")

        # Limit to max_items
        if args.max_items and len(df) > args.max_items:
            df = df.head(args.max_items)
            print(f"✓ Limited to {args.max_items} items")

        print()
        print("🚀 Starting ingestion...")
        print("=" * 80)
        print()

        # Track statistics
        successful = 0
        failed = 0

        # Note: In a real batch scenario, we might use the universal pipeline
        # or call the RAG instance directly.
        # For this example, we'll demonstrate using the universal pipeline
        # which already handles file processing.

        # If it's a CSV, the universal pipeline can handle it
        stats = run_universal_pipeline(
            input_path=str(dataset_path),
            max_items=args.max_items
        )

        print()
        print("=" * 80)
        print("✅ Ingestion Complete!")
        print("=" * 80)
        print(f"Total Files: {stats.get('total_files', 0)}")
        print(f"Successful: {stats.get('successful_files', 0)}")
        print(f"Failed: {stats.get('failed_files', 0)}")
        print()

        # Display storage statistics
        storage_dir = Path("./rag_storage")
        if storage_dir.exists():
            vdb_file = storage_dir / "vdb_chunks.json"
            if vdb_file.exists():
                import json
                with open(vdb_file) as f:
                    vdb_data = json.load(f)
                    print("📊 Storage Statistics:")
                    # Check for different possible key names in vdb_chunks.json
                    count = len(vdb_data.get('storage', vdb_data.get('matrix', [])))
                    print(f"  - Vector embeddings: {count}")

        print()
        print("You can now query the system:")
        print("  uv run ark query \"Your research question\"")

    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
