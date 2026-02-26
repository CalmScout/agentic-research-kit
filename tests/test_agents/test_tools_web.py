import pytest
import json
import httpx
from unittest.mock import MagicMock, patch, AsyncMock
from src.agents.tools.web import WebSearchTool, WebFetchTool, _strip_tags, _normalize, _validate_url

def test_strip_tags():
    html = "<html><body><h1>Title</h1><script>alert(1)</script><style>.css{}</style><p>Hello &amp; world</p></body></html>"
    stripped = _strip_tags(html)
    assert stripped == "TitleHello & world"

def test_normalize():
    text = "Hello    world\n\n\nNew line"
    normalized = _normalize(text)
    assert normalized == "Hello world\n\nNew line"

def test_validate_url():
    assert _validate_url("https://example.com")[0] is True
    assert _validate_url("http://localhost:8000")[0] is True
    assert _validate_url("ftp://example.com")[0] is False
    assert _validate_url("https://")[0] is False

@pytest.mark.asyncio
async def test_web_search_tool_success():
    tool = WebSearchTool(api_key="fake_key")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "web": {
            "results": [
                {"title": "Result 1", "url": "http://r1", "description": "Desc 1"},
                {"title": "Result 2", "url": "http://r2"}
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        result = await tool.execute(query="test query")
        
        assert "Web search results for: test query" in result
        assert "Result 1" in result
        assert "http://r1" in result
        assert "Desc 1" in result
        assert "Result 2" in result

@pytest.mark.asyncio
async def test_web_search_tool_no_key():
    with patch.dict("os.environ", {}, clear=True):
        tool = WebSearchTool(api_key=None)
        result = await tool.execute(query="test")
        assert "Error: BRAVE_API_KEY not configured" in result

@pytest.mark.asyncio
async def test_web_fetch_tool_success_html():
    tool = WebFetchTool()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = "<html><head><title>Test Title</title></head><body><p>Test content</p></body></html>"
    mock_response.url = "http://test.com"
    
    mock_doc = MagicMock()
    mock_doc.title.return_value = "Test Title"
    mock_doc.summary.return_value = "<p>Test content</p>"
    
    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
        patch("readability.Document", return_value=mock_doc)
    ):
        mock_get.return_value = mock_response
        
        result_json = await tool.execute(url="http://test.com")
        result = json.loads(result_json)
        
        assert result["url"] == "http://test.com"
        assert "Test Title" in result["text"]
        assert "Test content" in result["text"]
        assert result["extractor"] == "readability"

@pytest.mark.asyncio
async def test_web_fetch_tool_success_json():
    tool = WebFetchTool()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"key": "value"}
    mock_response.url = "http://test.com/api"
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        result_json = await tool.execute(url="http://test.com/api")
        result = json.loads(result_json)
        
        assert result["extractor"] == "json"
        assert '"key": "value"' in result["text"]

def test_web_fetch_to_markdown():
    tool = WebFetchTool()
    html = '<h1>Title</h1><p>Text with <a href="http://link">link</a></p><ul><li>Item 1</li></ul>'
    markdown = tool._to_markdown(html)
    assert "# Title" in markdown
    assert "Text with [link](http://link)" in markdown
    assert "- Item 1" in markdown
