"""Document loaders for multiple file formats.

Provides abstract base class and concrete implementations for loading
different document types (PDF, DOCX, HTML, TXT) into a unified format.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Abstract base class for document loaders.

    All document loaders must implement the load() method which returns
    a list of dictionaries with 'content' and 'metadata' fields.
    """

    @abstractmethod
    def load(self, path: str) -> list[dict[str, Any]]:
        """Load document and return structured content.

        Args:
            path: Path to the document file

        Returns:
            List of dictionaries with:
                - 'content': The text content
                - 'metadata': Dict with source, page/section info, type, etc.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    def _validate_path(self, path: str) -> Path:
        """Validate that file exists and return Path object.

        Args:
            path: File path string

        Returns:
            Path object

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p


class PDFLoader(DocumentLoader):
    """Load PDF documents using PyMuPDF (fitz).

    Extracts text from each page with metadata including page number.
    Handles encrypted PDFs and various PDF formats.
    """

    def __init__(self, extract_images: bool = False):
        """Initialize PDF loader.

        Args:
            extract_images: Whether to extract images (not implemented yet)
        """
        self.extract_images = extract_images
        try:
            import fitz  # PyMuPDF

            self.fitz = fitz
        except ImportError as e:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF loading. " "Install with: uv add pymupdf"
            ) from e

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load PDF and extract text from each page.

        Args:
            path: Path to PDF file

        Returns:
            List of documents, one per page
        """
        path_obj = self._validate_path(path)
        docs = []

        try:
            pdf = self.fitz.open(str(path_obj))

            for page_num, page in enumerate(pdf):
                text = page.get_text()

                # Skip empty pages
                if not text.strip():
                    continue

                doc = {
                    "content": text,
                    "metadata": {
                        "source": str(path_obj),
                        "filename": path_obj.name,
                        "page": page_num + 1,
                        "total_pages": len(pdf),
                        "type": "pdf",
                    },
                }
                docs.append(doc)

            pdf.close()
            logger.info(f"Loaded {len(docs)} pages from {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load PDF {path}: {e}") from e

        return docs


class DocxLoader(DocumentLoader):
    """Load Word documents (.docx) using python-docx.

    Extracts text from paragraphs with document metadata.
    """

    def __init__(self):
        """Initialize DOCX loader."""
        try:
            from docx import Document

            self.Document = Document
        except ImportError as e:
            raise ImportError(
                "python-docx is required for DOCX loading. " "Install with: uv add python-docx"
            ) from e

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load DOCX and extract paragraphs.

        Args:
            path: Path to DOCX file

        Returns:
            List of documents (grouped by paragraphs or sections)
        """
        path_obj = self._validate_path(path)
        docs = []

        try:
            doc = self.Document(str(path_obj))

            # Extract paragraphs, grouping them into reasonable chunks
            current_chunk = []
            chunk_size = 0
            max_chunk_size = 2000  # Characters per chunk

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue

                current_chunk.append(text)
                chunk_size += len(text)

                # Create new chunk when size limit reached
                if chunk_size >= max_chunk_size:
                    chunk_text = "\n\n".join(current_chunk)
                    docs.append(
                        {
                            "content": chunk_text,
                            "metadata": {
                                "source": str(path_obj),
                                "filename": path_obj.name,
                                "type": "docx",
                            },
                        }
                    )
                    current_chunk = []
                    chunk_size = 0

            # Add remaining paragraphs
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                docs.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "source": str(path_obj),
                            "filename": path_obj.name,
                            "type": "docx",
                        },
                    }
                )

            logger.info(f"Loaded {len(docs)} chunks from {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load DOCX {path}: {e}") from e

        return docs


class HTMLLoader(DocumentLoader):
    """Load HTML documents using BeautifulSoup.

    Extracts main text content from HTML files, removing scripts and styles.
    """

    def __init__(self):
        """Initialize HTML loader."""
        try:
            from bs4 import BeautifulSoup

            self.BeautifulSoup = BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "beautifulsoup4 is required for HTML loading. "
                "Install with: uv add beautifulsoup4 lxml"
            ) from e

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load HTML and extract text content.

        Args:
            path: Path to HTML file

        Returns:
            List with single document containing extracted text
        """
        path_obj = self._validate_path(path)
        docs = []

        try:
            with open(path_obj, encoding="utf-8") as f:
                html = f.read()

            soup = self.BeautifulSoup(html, "lxml")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            docs.append(
                {
                    "content": text,
                    "metadata": {
                        "source": str(path_obj),
                        "filename": path_obj.name,
                        "title": soup.title.string if soup.title else path_obj.name,
                        "type": "html",
                    },
                }
            )

            logger.info(f"Loaded HTML from {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load HTML {path}: {e}") from e

        return docs


class TextLoader(DocumentLoader):
    """Load plain text files.

    Simple loader for .txt and other plain text formats.
    """

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load text file.

        Args:
            path: Path to text file

        Returns:
            List with single document containing text content
        """
        path_obj = self._validate_path(path)
        docs = []

        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                with open(path_obj, encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(path_obj, encoding="latin-1") as f:
                    text = f.read()

            docs.append(
                {
                    "content": text,
                    "metadata": {
                        "source": str(path_obj),
                        "filename": path_obj.name,
                        "type": "text",
                    },
                }
            )

            logger.info(f"Loaded text file {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load text file {path}: {e}") from e

        return docs


class MarkdownLoader(DocumentLoader):
    """Load Markdown files.

    Simple loader for .md files that preserves structure.
    """

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load Markdown file.

        Args:
            path: Path to Markdown file

        Returns:
            List with single document containing markdown content
        """
        path_obj = self._validate_path(path)
        docs = []

        try:
            with open(path_obj, encoding="utf-8") as f:
                text = f.read()

            docs.append(
                {
                    "content": text,
                    "metadata": {
                        "source": str(path_obj),
                        "filename": path_obj.name,
                        "type": "markdown",
                    },
                }
            )

            logger.info(f"Loaded Markdown file {path_obj.name}")

        except Exception as e:
            raise ValueError(f"Failed to load Markdown file {path}: {e}") from e

        return docs


# Loader registry mapping file extensions to loader classes
LOADER_REGISTRY: dict[str, type] = {
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".doc": DocxLoader,  # python-docx only supports .docx, but we map it anyway
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
}


def get_loader(file_path: str) -> DocumentLoader:
    """Get appropriate loader for file based on extension.

    Args:
        file_path: Path to file

    Returns:
        DocumentLoader instance

    Raises:
        ValueError: If file type is not supported
    """
    ext = Path(file_path).suffix.lower()

    if ext not in LOADER_REGISTRY:
        supported = ", ".join(LOADER_REGISTRY.keys())
        raise ValueError(f"Unsupported file type: {ext}. " f"Supported types: {supported}")

    loader_class = LOADER_REGISTRY[ext]
    return cast(DocumentLoader, loader_class())


def load_document(file_path: str) -> list[dict[str, Any]]:
    """Load a document using the appropriate loader.

    Convenience function that automatically selects the right loader
    based on file extension.

    Args:
        file_path: Path to document file

    Returns:
        List of document dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported or file is invalid
    """
    loader = get_loader(file_path)
    return loader.load(file_path)
