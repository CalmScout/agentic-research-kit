from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from src.data_ingestion.image_downloader import AsyncImageDownloader


@pytest.fixture
def downloader(tmp_path):
    return AsyncImageDownloader(output_dir=str(tmp_path))

@pytest.mark.asyncio
async def test_download_with_retries_success(downloader):
    mock_session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"Content-Type": "image/jpeg"}
    mock_response.read.return_value = b"image data"
    mock_response.__aenter__.return_value = mock_response
    mock_session.get.return_value = mock_response

    with patch("builtins.open", mock_open()) as mock_file:
        path = await downloader._download_with_retries(mock_session, "http://test.com/img.jpg", 1)

        assert "claim_1.jpg" in path
        mock_file().write.assert_called_with(b"image data")

@pytest.mark.asyncio
async def test_download_with_retries_failure_then_wayback(downloader):
    mock_session = MagicMock()

    # Fail direct download
    mock_fail_response = AsyncMock()
    mock_fail_response.status = 404
    mock_fail_response.__aenter__.return_value = mock_fail_response

    # Success on wayback
    mock_success_response = AsyncMock()
    mock_success_response.status = 200
    mock_success_response.headers = {"Content-Type": "image/png"}
    mock_success_response.read.return_value = b"wayback data"
    mock_success_response.__aenter__.return_value = mock_success_response

    mock_session.get.side_effect = [mock_fail_response, mock_fail_response, mock_fail_response, mock_success_response]

    with patch("builtins.open", mock_open()) as mock_file:
        path = await downloader._download_with_retries(mock_session, "http://test.com/img.png", 2)

        assert "claim_2_wayback.png" in path
        mock_file().write.assert_called_with(b"wayback data")

def test_get_extension(downloader):
    assert downloader._get_extension("image/jpeg") == ".jpg"
    assert downloader._get_extension("image/png") == ".png"
    assert downloader._get_extension("image/webp") == ".webp"
    assert downloader._get_extension("unknown") == ".jpg"

@pytest.mark.asyncio
async def test_download_batch(downloader):
    urls = ["http://u1", "http://u2"]
    ids = [10, 20]

    # Mock download_single to return dummy paths
    with patch.object(AsyncImageDownloader, "download_single", new_callable=AsyncMock) as mock_download:
        mock_download.side_effect = ["/path/10.jpg", "/path/20.png"]

        image_map = await downloader.download_batch(urls, ids)

        assert image_map[10] == "/path/10.jpg"
        assert image_map[20] == "/path/20.png"
        assert len(image_map) == 2
