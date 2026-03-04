import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_ingestion.document_loaders import (
    DocxLoader,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    get_loader,
    load_document,
)


@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"test content")
        temp_path = tf.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_validate_path_exists(temp_file):
    loader = TextLoader()
    path = loader._validate_path(temp_file)
    assert isinstance(path, Path)
    assert str(path) == temp_file

def test_validate_path_not_exists():
    loader = TextLoader()
    with pytest.raises(FileNotFoundError):
        loader._validate_path("non_existent_file.txt")

def test_text_loader(temp_file):
    loader = TextLoader()
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("hello world")

    docs = loader.load(temp_file)
    assert len(docs) == 1
    assert docs[0]["content"] == "hello world"
    assert docs[0]["metadata"]["type"] == "text"

def test_markdown_loader(temp_file):
    loader = MarkdownLoader()
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("# heading\ncontent")

    docs = loader.load(temp_file)
    assert len(docs) == 1
    assert docs[0]["content"] == "# heading\ncontent"
    assert docs[0]["metadata"]["type"] == "markdown"

@patch("fitz.open")
def test_pdf_loader(mock_fitz_open, temp_file):
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "page text"
    mock_pdf.__iter__.return_value = [mock_page]
    mock_pdf.__len__.return_value = 1
    mock_fitz_open.return_value = mock_pdf

    loader = PDFLoader()
    docs = loader.load(temp_file)

    assert len(docs) == 1
    assert docs[0]["content"] == "page text"
    assert docs[0]["metadata"]["page"] == 1
    assert docs[0]["metadata"]["type"] == "pdf"

@patch("docx.Document")
def test_docx_loader(mock_docx, temp_file):
    mock_doc = MagicMock()
    para = MagicMock()
    para.text = "paragraph text"
    mock_doc.paragraphs = [para]
    mock_docx.return_value = mock_doc

    loader = DocxLoader()
    docs = loader.load(temp_file)

    assert len(docs) == 1
    assert docs[0]["content"] == "paragraph text"
    assert docs[0]["metadata"]["type"] == "docx"

@patch("bs4.BeautifulSoup")
def test_html_loader(mock_bs, temp_file):
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>title</h1><p>text</p></body></html>")

    mock_soup = MagicMock()
    mock_soup.get_text.return_value = "title\ntext"
    mock_soup.title.string = "Test Title"
    mock_bs.return_value = mock_soup

    loader = HTMLLoader()
    docs = loader.load(temp_file)

    assert len(docs) == 1
    assert docs[0]["content"] == "title\ntext"
    assert docs[0]["metadata"]["type"] == "html"
    assert docs[0]["metadata"]["title"] == "Test Title"

def test_get_loader():
    assert isinstance(get_loader("test.pdf"), PDFLoader)
    assert isinstance(get_loader("test.docx"), DocxLoader)
    assert isinstance(get_loader("test.txt"), TextLoader)
    assert isinstance(get_loader("test.html"), HTMLLoader)
    assert isinstance(get_loader("test.md"), MarkdownLoader)

    with pytest.raises(ValueError, match="Unsupported file type"):
        get_loader("test.unknown")

@patch("src.data_ingestion.document_loaders.TextLoader.load")
def test_load_document(mock_load, temp_file):
    mock_load.return_value = [{"content": "data", "metadata": {}}]
    docs = load_document(temp_file + ".txt")
    assert docs[0]["content"] == "data"
