"""Universal ingestion pipeline for multiple document types.

Orchestrates the ingestion of documents from various sources (PDF, DOCX, HTML, TXT, CSV)
into the RAG system. Provides a generic interface that works with different file types.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from tqdm import tqdm

from ..utils.logger import configure_logging
from .document_loaders import LOADER_REGISTRY, load_document
from .generic_rag_ingester import GenericRAGIngester

logger = logging.getLogger(__name__)


class UniversalIngestionPipeline:
    """Generic ingestion pipeline for multiple document types.

    Supports PDF, DOCX, HTML, TXT, and CSV files with automatic
    format detection and processing.
    """

    def __init__(
        self,
        working_dir: str,
        use_gpu: bool = True,
        content_template: str | None = None,
        metadata_fields: list[str] | None = None,
    ):
        """Initialize the universal ingestion pipeline.

        Args:
            working_dir: Directory for RAG storage
            use_gpu: Whether to use GPU acceleration for embeddings
            content_template: Template for formatting content (default: "{content}")
            metadata_fields: List of fields to include in metadata
        """
        self.working_dir = Path(working_dir)
        self.use_gpu = use_gpu
        self.supported_formats = list(LOADER_REGISTRY.keys())

        # Default template for document content
        if content_template is None:
            content_template = "Title: {title}\nContent: {content}"
            if metadata_fields is None:
                metadata_fields = ["title", "source", "page"]

        self.content_template = content_template
        self.metadata_fields = metadata_fields or []

        logger.info("Initialized UniversalIngestionPipeline")
        logger.info(f"Working directory: {self.working_dir}")
        logger.info(f"Supported formats: {', '.join(self.supported_formats)}")

    async def ingest_files(
        self, file_paths: list[str], max_items: int | None = None, show_progress: bool = True
    ) -> dict[str, Any]:
        """Ingest specific list of files.

        Args:
            file_paths: List of file paths to ingest
            max_items: Maximum number of files to process (for testing)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with ingestion statistics
        """
        # Configure logging
        configure_logging()

        logger.info("=" * 80)
        logger.info("STARTING UNIVERSAL INGESTION PIPELINE")
        logger.info("=" * 80)

        # Limit files if max_items specified
        if max_items is not None:
            file_paths = file_paths[:max_items]
            logger.info(f"🧪 Test mode: Limited to {max_items} files")

        logger.info(f"📁 Files to process: {len(file_paths)}")

        # Stage 1: Load documents
        logger.info("\n[Stage 1/2] Loading documents...")
        all_docs = []
        failed_files = []

        files_iter: Any = (
            file_paths if not show_progress else tqdm(file_paths, desc="Loading files")
        )

        for file_path in files_iter:
            try:
                docs = load_document(file_path)
                all_docs.extend(docs)
                if show_progress:
                    files_iter.write(f"✓ Loaded {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                failed_files.append({"file": file_path, "error": str(e)})

        logger.info(
            f"✓ Loaded {len(all_docs)} chunks from {len(file_paths) - len(failed_files)} files"
        )
        if failed_files:
            logger.warning(f"✗ Failed to load {len(failed_files)} files")

        # Stage 2: Ingest into RAG
        logger.info("\n[Stage 2/2] Ingesting into RAG system...")

        ingester = GenericRAGIngester(
            working_dir=str(self.working_dir),
            use_gpu=self.use_gpu,
            content_template=self.content_template,
            content_fields=["title", "content", "source", "page"],  # Include title for formatting
            metadata_fields=self.metadata_fields,
        )

        try:
            # Convert docs to format expected by RAG ingester
            # The RAG ingester expects a DataFrame with specific columns for CSV data,
            # but for universal ingestion, we need to adapt
            ingest_stats = await self._ingest_documents(ingester, all_docs)

            logger.info("✓ Ingestion complete!")
            logger.info(f"  - Items ingested: {ingest_stats.get('total_items', 0)}")

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

        # Final statistics
        final_stats = {
            "total_files": len(file_paths),
            "successful_files": len(file_paths) - len(failed_files),
            "failed_files": len(failed_files),
            "total_chunks": len(all_docs),
            "ingest_stats": ingest_stats,
            "failed_details": failed_files,
        }

        logger.info("\n" + "=" * 80)
        logger.info("INGESTION PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total files: {final_stats['total_files']}")
        logger.info(f"Successful: {final_stats['successful_files']}")
        logger.info(f"Failed: {final_stats['failed_files']}")
        logger.info(f"Total chunks: {final_stats['total_chunks']}")
        logger.info(
            f"Items ingested: {cast(Any, final_stats['ingest_stats']).get('total_items', 0)}"
        )

        return final_stats

    async def ingest_directory(
        self,
        dir_path: str,
        pattern: str = "**/*.pdf",
        recursive: bool = True,
        max_items: int | None = None,
    ) -> dict[str, Any]:
        """Ingest all documents matching pattern from a directory.

        Args:
            dir_path: Path to directory containing documents
            pattern: Glob pattern for matching files (e.g., "**/*.pdf", "*.docx")
            recursive: Whether to search recursively
            max_items: Maximum number of files to process

        Returns:
            Dictionary with ingestion statistics
        """
        dir_path_obj = Path(dir_path)

        if not dir_path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path_obj.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        # Find matching files

        # Use rglob for recursive, glob for non-recursive
        if recursive:
            matching_files = list(dir_path_obj.rglob(pattern.split("/")[-1]))
        else:
            matching_files = list(dir_path_obj.glob(pattern))

        # Filter by supported extensions
        supported_files = [
            str(f)
            for f in matching_files
            if f.is_file() and f.suffix.lower() in self.supported_formats
        ]

        if not supported_files:
            logger.warning(f"No supported files found in {dir_path}")
            return {
                "total_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "ingest_stats": {},
                "failed_details": [],
            }

        logger.info(f"Found {len(supported_files)} files matching pattern")

        return await self.ingest_files(file_paths=supported_files, max_items=max_items)

    async def _ingest_documents(
        self, ingester: GenericRAGIngester, docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Ingest loaded documents into RAG system.

        Args:
            ingester: Generic RAG ingester instance
            docs: List of document dictionaries

        Returns:
            Ingestion statistics
        """
        # Convert documents to DataFrame format expected by GenericRAGIngester
        df_data = []

        for doc in cast(list[dict[str, Any]], docs):
            row = {
                "title": cast(Any, doc.get("metadata", {})).get("title", "Untitled"),
                "content": doc.get("content", ""),
                "source": cast(Any, doc.get("metadata", {})).get("source", ""),
                "page": cast(Any, doc.get("metadata", {})).get("page", 1),
                "doc_type": cast(Any, doc.get("metadata", {})).get("doc_type", "unknown"),
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        logger.info(f"Prepared {len(df)} document chunks for ingestion")

        # Ingest using GenericRAGIngester
        stats = await ingester.ingest_df(df, id_column="title")

        return stats


def run_sync_universal_ingestion(
    file_paths: list[str],
    working_dir: str = "./rag_storage",
    use_gpu: bool = True,
    max_items: int | None = None,
) -> dict:
    """Run universal ingestion pipeline synchronously (for CLI).

    Args:
        file_paths: List of file paths to ingest
        working_dir: Directory for RAG storage
        use_gpu: Whether to use GPU acceleration
        max_items: Maximum number of files to process

    Returns:
        Pipeline statistics dictionary
    """
    pipeline = UniversalIngestionPipeline(working_dir=working_dir, use_gpu=use_gpu)
    return asyncio.run(pipeline.ingest_files(file_paths, max_items=max_items))


def run_sync_directory_ingestion(
    dir_path: str,
    pattern: str = "**/*.pdf",
    working_dir: str = "./rag_storage",
    use_gpu: bool = True,
    recursive: bool = True,
    max_items: int | None = None,
) -> dict:
    """Run directory ingestion pipeline synchronously (for CLI).

    Args:
        dir_path: Path to directory containing documents
        pattern: Glob pattern for matching files
        working_dir: Directory for RAG storage
        use_gpu: Whether to use GPU acceleration
        recursive: Whether to search recursively
        max_items: Maximum number of files to process

    Returns:
        Pipeline statistics dictionary
    """
    pipeline = UniversalIngestionPipeline(working_dir=working_dir, use_gpu=use_gpu)
    return asyncio.run(
        pipeline.ingest_directory(
            dir_path=dir_path, pattern=pattern, recursive=recursive, max_items=max_items
        )
    )
