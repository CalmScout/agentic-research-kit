import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import subprocess
import httpx
from src.agents.lightrag_client import LightRAGHTTPClient, get_lightrag_client
from src.utils.config import Settings

@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.lightrag_api_host = "localhost"
    settings.lightrag_api_port = 8000
    settings.rag_working_dir = "/tmp/rag"
    settings.lightrag_auto_start_server = True
    return settings

@pytest.fixture
def client(mock_settings):
    return LightRAGHTTPClient(settings=mock_settings)

@pytest.mark.asyncio
async def test_health_check_success(client):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        
        result = await client.health_check()
        
        assert result is True
        mock_get.assert_called_once_with("http://localhost:8000/health")

@pytest.mark.asyncio
async def test_health_check_failure(client):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Connection failed")
        
        result = await client.health_check()
        
        assert result is False

@pytest.mark.asyncio
async def test_start_server(client):
    with (
        patch("subprocess.Popen") as mock_popen,
        patch.object(LightRAGHTTPClient, "health_check", new_callable=AsyncMock) as mock_health
    ):
        
        mock_health.side_effect = [False, True]
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        with patch("asyncio.sleep", return_value=None):
            await client.start_server()
        
        assert mock_popen.called
        assert client._server_process == mock_process

@pytest.mark.asyncio
async def test_query_hybrid_success(client):
    mock_response_data = {
        "data": {
            "chunks": [
                {"content": "chunk 1", "chunk_id": "1", "file_path": "file1.txt", "reference_id": "ref1"},
                {"content": "chunk 2", "chunk_id": "2", "file_path": "file2.txt", "reference_id": "ref2"}
            ]
        }
    }
    
    with (
        patch.object(LightRAGHTTPClient, "ensure_server_running", new_callable=AsyncMock),
        patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post
    ):
        
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = await client.query_hybrid("test query", top_k=10)
        
        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_docs"][0]["text"] == "chunk 1"
        assert result["retrieval_method"] == "hybrid"

@pytest.mark.asyncio
async def test_stop_server(client):
    mock_process = MagicMock()
    client._server_process = mock_process
    
    await client.stop_server()
    
    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()
    assert client._server_process is None

def test_get_lightrag_client():
    client1 = get_lightrag_client()
    client2 = get_lightrag_client()
    assert client1 is client2
    assert isinstance(client1, LightRAGHTTPClient)
