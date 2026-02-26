import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from src.data_ingestion.universal_pipeline import UniversalIngestionPipeline

@pytest.fixture
def pipeline():
    return UniversalIngestionPipeline(working_dir="./test_rag")

@pytest.mark.asyncio
@patch("src.data_ingestion.universal_pipeline.load_document")
@patch("src.data_ingestion.universal_pipeline.GenericRAGIngester")
async def test_ingest_files(mock_ingester_class, mock_load, pipeline):
    mock_load.return_value = [{"content": "text", "metadata": {"title": "t1"}}]
    
    mock_ingester = MagicMock()
    mock_ingester.ingest_df = AsyncMock(return_value={"total_items": 1})
    mock_ingester_class.return_value = mock_ingester
    
    stats = await pipeline.ingest_files(["test.pdf"])
    
    assert stats["total_files"] == 1
    assert stats["successful_files"] == 1
    assert stats["total_chunks"] == 1
    assert stats["ingest_stats"]["total_items"] == 1
    
    mock_load.assert_called_with("test.pdf")
    mock_ingester.ingest_df.assert_called()

@pytest.mark.asyncio
@patch("src.data_ingestion.universal_pipeline.UniversalIngestionPipeline.ingest_files")
async def test_ingest_directory(mock_ingest_files, pipeline, tmp_path):
    # Create dummy files
    (tmp_path / "test1.pdf").touch()
    (tmp_path / "test2.txt").touch()
    (tmp_path / "test3.unknown").touch()
    
    mock_ingest_files.return_value = {"total_files": 2}
    
    stats = await pipeline.ingest_directory(str(tmp_path), pattern="*")
    
    assert mock_ingest_files.called
    # matching_files should include test1.pdf and test2.txt (if pattern is * and they match supported_formats)
    args, kwargs = mock_ingest_files.call_args
    file_paths = kwargs.get("file_paths") or args[0]
    assert any("test1.pdf" in f for f in file_paths)
    assert any("test2.txt" in f for f in file_paths)
    assert not any("test3.unknown" in f for f in file_paths)

@pytest.mark.asyncio
@patch("src.data_ingestion.universal_pipeline.GenericRAGIngester")
async def test_ingest_documents_helper(mock_ingester_class, pipeline):
    mock_ingester = MagicMock()
    mock_ingester.ingest_df = AsyncMock(return_value={"total_items": 2})
    
    docs = [
        {"content": "c1", "metadata": {"title": "t1"}},
        {"content": "c2", "metadata": {"title": "t2"}}
    ]
    
    stats = await pipeline._ingest_documents(mock_ingester, docs)
    
    assert stats["total_items"] == 2
    args, kwargs = mock_ingester.ingest_df.call_args
    df = args[0]
    assert len(df) == 2
    assert list(df["title"]) == ["t1", "t2"]
