"""
Example: Custom Data Source Integration

This example demonstrates how to:
1. Create a custom document loader for specialized data sources
2. Register the loader with the universal ingestion pipeline
3. Ingest documents from custom sources

Use Case: When you have documents in a format not natively supported by ARK.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_ingestion.document_loaders import LOADER_REGISTRY, DocumentLoader
from data_ingestion.universal_pipeline import UniversalIngestionPipeline

# ============================================================================
# Example 1: Custom JSON Document Loader
# ============================================================================

class JSONDataLoader(DocumentLoader):
    """Load documents from JSON files with custom structure.

    Example JSON structure:
    [
        {
            "id": "doc1",
            "title": "Document Title",
            "content": "Document content here...",
            "metadata": {
                "author": "John Doe",
                "date": "2024-01-01",
                "category": "research"
            }
        },
        ...
    ]
    """

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load JSON documents."""
        import json

        path_obj = self._validate_path(path)
        docs = []

        try:
            with open(path_obj, encoding='utf-8') as f:
                data = json.load(f)

            # Handle both list and single object
            items = data if isinstance(data, list) else [data]

            for item in items:
                content = item.get("content", "")
                if not content:
                    continue

                doc = {
                    "content": content,
                    "metadata": {
                        "source": str(path_obj),
                        "filename": path_obj.name,
                        "type": "json",
                        "id": item.get("id"),
                        "title": item.get("title"),
                        **item.get("metadata", {})
                    }
                }
                docs.append(doc)

            print(f"✓ Loaded {len(docs)} documents from {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load JSON {path}: {e}")

        return docs


# ============================================================================
# Example 2: Custom Markdown Loader with Frontmatter
# ============================================================================

class MarkdownWithFrontmatterLoader(DocumentLoader):
    """Load Markdown files with YAML frontmatter.

    Example Markdown structure:
    ---
    title: Document Title
    author: Jane Doe
    date: 2024-01-01
    tags: [research, ai]
    ---
    # Document Content

    The document content goes here...
    """

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load Markdown with frontmatter."""
        path_obj = self._validate_path(path)
        docs = []

        try:
            with open(path_obj, encoding='utf-8') as f:
                content = f.read()

            # Parse frontmatter
            frontmatter = {}
            if content.startswith('---'):
                _, frontmatter_text, content = content.split('---', 2)
                # Parse YAML frontmatter (simplified)
                for line in frontmatter_text.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip()

            doc = {
                "content": content.strip(),
                "metadata": {
                    "source": str(path_obj),
                    "filename": path_obj.name,
                    "type": "markdown",
                    **frontmatter
                }
            }
            docs.append(doc)

            print(f"✓ Loaded Markdown document: {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load Markdown {path}: {e}")

        return docs


# ============================================================================
# Example 3: Register and Use Custom Loaders
# ============================================================================

async def example_custom_loaders():
    """Demonstrate custom loader registration and usage."""

    print("=" * 80)
    print("Custom Data Source Integration Examples")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Example 1: Register Custom JSON Loader
    # -----------------------------------------------------------------------
    print("\n📝 Example 1: Custom JSON Loader")

    # Register the custom loader
    LOADER_REGISTRY[".json"] = JSONDataLoader
    LOADER_REGISTRY[".jsonl"] = JSONDataLoader  # Also support JSONL

    print("   ✓ Registered JSON loader for .json and .jsonl files")

    # Use it
    json_path = "./data/custom_documents.json"
    if Path(json_path).exists():
        from data_ingestion.document_loaders import load_document

        docs = load_document(json_path)
        print(f"   ✓ Loaded {len(docs)} documents from JSON")
    else:
        print(f"   ⚠️  File not found: {json_path}")

    # -----------------------------------------------------------------------
    # Example 2: Register Custom Markdown Loader
    # -----------------------------------------------------------------------
    print("\n📝 Example 2: Custom Markdown Loader with Frontmatter")

    LOADER_REGISTRY[".md"] = MarkdownWithFrontmatterLoader

    print("   ✓ Registered custom Markdown loader for .md files")

    # Use it
    md_path = "./data/document_with_frontmatter.md"
    if Path(md_path).exists():
        from data_ingestion.document_loaders import load_document

        docs = load_document(md_path)
        print(f"   ✓ Loaded {len(docs)} documents from Markdown")
    else:
        print(f"   ⚠️  File not found: {md_path}")

    # -----------------------------------------------------------------------
    # Example 3: Ingest Directory with Custom Loaders
    # -----------------------------------------------------------------------
    print("\n📝 Example 3: Batch Ingestion with Custom Loaders")

    pipeline = UniversalIngestionPipeline(
        working_dir="./rag_storage/custom_data",
        use_gpu=True
    )

    custom_data_dir = "./data/custom_documents"
    if Path(custom_data_dir).exists():
        stats = await pipeline.ingest_directory(
            dir_path=custom_data_dir,
            pattern="**/*.{json,jsonl,md}",
            recursive=True
        )

        print("\n✅ Ingestion Complete:")
        print(f"   - Files processed: {stats['successful_files']}")
        print(f"   - Total chunks: {stats['total_chunks']}")
        print(f"   - Items ingested: {stats['ingest_stats']['total_items']}")
    else:
        print(f"   ⚠️  Directory not found: {custom_data_dir}")

    print("\n" + "=" * 80)
    print("Custom Loader Examples Complete!")
    print("=" * 80)


# ============================================================================
# Example 4: API Data Loader
# ============================================================================

class APIResponseLoader(DocumentLoader):
    """Load documents from API responses (saved as JSON).

    This loader handles API responses that have been saved to files.
    """

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load API response documents."""
        import json

        path_obj = self._validate_path(path)
        docs = []

        try:
            with open(path_obj, encoding='utf-8') as f:
                data = json.load(f)

            # Handle different API response structures
            if isinstance(data, dict):
                # Structure: {"results": [...], "metadata": {...}}
                items = data.get("results", data.get("data", []))
                api_metadata = {k: v for k, v in data.items() if k not in ["results", "data"]}
            else:
                items = data
                api_metadata = {}

            for item in items:
                # Extract content from various possible fields
                content = (
                    item.get("content") or
                    item.get("text") or
                    item.get("body") or
                    str(item)
                )

                doc = {
                    "content": content,
                    "metadata": {
                        "source": str(path_obj),
                        "filename": path_obj.name,
                        "type": "api_response",
                        "api_metadata": api_metadata,
                        **{k: v for k, v in item.items() if k not in ["content", "text", "body"]}
                    }
                }
                docs.append(doc)

            print(f"✓ Loaded {len(docs)} API response documents from {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load API response {path}: {e}")

        return docs


async def example_api_loader():
    """Demonstrate API response loader."""

    print("\n" + "=" * 80)
    print("API Response Loader Example")
    print("=" * 80)

    # Register the API response loader
    LOADER_REGISTRY[".api.json"] = APIResponseLoader

    print("✓ Registered API response loader for .api.json files")

    # Example usage
    api_response_path = "./data/api_response.api.json"
    if Path(api_response_path).exists():
        from data_ingestion.document_loaders import load_document

        docs = load_document(api_response_path)
        print(f"✓ Loaded {len(docs)} documents from API response")
    else:
        print(f"⚠️  File not found: {api_response_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   Agentic Research Kit: Custom Data Source Integration                       ║
║                                                                              ║
║   This example demonstrates how to create and use custom document          ║
║   loaders for specialized data sources not natively supported.              ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run examples
    asyncio.run(example_custom_loaders())
    asyncio.run(example_api_loader())
