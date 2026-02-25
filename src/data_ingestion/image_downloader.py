"""Async image downloader with Wayback Machine fallback."""

import asyncio
import logging
from pathlib import Path

import aiohttp
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AsyncImageDownloader:
    """Download claim review images asynchronously with fallback."""

    def __init__(
        self,
        output_dir: str = "./data/images",
        timeout: int = 10,
        max_workers: int = 10,
        max_retries: int = 3,
    ):
        """Initialize downloader.

        Args:
            output_dir: Directory to save images
            timeout: Request timeout in seconds
            max_workers: Maximum concurrent downloads
            max_retries: Number of retry attempts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_workers = max_workers
        self.max_retries = max_retries

        # Track failed downloads
        self.failed_urls: list[str] = []

    async def download_single(
        self,
        session: aiohttp.ClientSession,
        url: str,
        claim_id: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        """Download single image with fallback.

        Args:
            session: aiohttp session
            url: Image URL
            claim_id: Claim ID for naming
            semaphore: Optional semaphore for concurrency limiting

        Returns:
            Local image path if successful, None otherwise
        """
        # Use semaphore if provided
        if semaphore:
            async with semaphore:
                return await self._download_with_retries(session, url, claim_id)
        else:
            return await self._download_with_retries(session, url, claim_id)

    async def _download_with_retries(
        self, session: aiohttp.ClientSession, url: str, claim_id: int
    ) -> str | None:
        """Download image with retries and Wayback fallback.

        Args:
            session: aiohttp session
            url: Image URL
            claim_id: Claim ID for naming

        Returns:
            Local image path if successful, None otherwise
        """
        # Check for existing files (simple caching)
        # Try common extensions
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
            existing_path = self.output_dir / f"claim_{claim_id}{ext}"
            if existing_path.exists():
                logger.debug(f"⊗ Skipped (already cached): {existing_path.name}")
                return str(existing_path)

        # Attempt 1: Direct download with retries
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "")
                        ext = self._get_extension(content_type)
                        path = self.output_dir / f"claim_{claim_id}{ext}"

                        with open(path, "wb") as f:
                            f.write(await response.read())

                        logger.debug(f"✓ Downloaded: {url}")
                        return str(path)
            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt < self.max_retries - 1:
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} for {url}")
                    await asyncio.sleep(1)  # Wait before retry
                else:
                    logger.debug(f"Direct download failed: {url} - {e}")

        # Attempt 2: Wayback Machine fallback
        wayback_path = await self._wayback_fallback(session, url, claim_id)
        if wayback_path:
            return wayback_path

        # Both attempts failed
        self.failed_urls.append(url)
        logger.warning(f"✗ Failed to download: {url}")
        return None

    async def _wayback_fallback(
        self, session: aiohttp.ClientSession, url: str, claim_id: int
    ) -> str | None:
        """Try Wayback Machine if direct download fails.

        Args:
            session: aiohttp session
            url: Original image URL
            claim_id: Claim ID for naming

        Returns:
            Local image path if successful, None otherwise
        """
        wayback_url = f"https://web.archive.org/web/{url}"

        try:
            async with session.get(wayback_url, timeout=self.timeout) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    ext = self._get_extension(content_type)
                    path = self.output_dir / f"claim_{claim_id}_wayback{ext}"

                    with open(path, "wb") as f:
                        f.write(await response.read())

                    logger.info(f"✓ Retrieved from Wayback: {url}")
                    return str(path)
        except Exception as e:
            logger.debug(f"Wayback fallback failed: {url} - {e}")

        return None

    async def download_batch(self, urls: list[str], claim_ids: list[int]) -> dict[int, str]:
        """Download batch of images.

        Args:
            urls: List of image URLs
            claim_ids: List of claim IDs

        Returns:
            Mapping of claim_id -> local_path
        """
        logger.info(f"Starting batch download: {len(urls)} images")
        logger.info(f"Workers: {self.max_workers}, Timeout: {self.timeout}s")

        # Create semaphore for concurrency limiting
        semaphore = asyncio.Semaphore(self.max_workers)

        # Create HTTP session
        async with aiohttp.ClientSession() as session:
            # Create tasks
            tasks = [
                self.download_single(session, url, cid, semaphore)
                for url, cid in zip(urls, claim_ids, strict=False)
            ]

            # Download with progress bar
            results = []
            with tqdm(total=len(tasks), desc="Downloading images", unit="img") as pbar:
                for future in asyncio.as_completed(tasks):
                    result = await future
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix(
                        success=sum(1 for r in results if r),
                        failed=sum(1 for r in results if not r),
                    )

        # Build mapping: claim_id -> local_path
        success_count = 0
        wayback_count = 0
        image_map = {}

        for cid, path in zip(claim_ids, results, strict=False):
            if path:
                image_map[cid] = path
                success_count += 1
                if "_wayback" in path:
                    wayback_count += 1

        logger.info(
            f"Download complete: {success_count} successful ({wayback_count} from Wayback), {len(self.failed_urls)} failed"
        )

        return image_map

    def _get_extension(self, content_type: str) -> str:
        """Get file extension from content-type.

        Args:
            content_type: Content-Type header

        Returns:
            File extension with leading dot
        """
        content_type = content_type.lower()

        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        elif "png" in content_type:
            return ".png"
        elif "webp" in content_type:
            return ".webp"
        elif "gif" in content_type:
            return ".gif"
        elif "bmp" in content_type:
            return ".bmp"
        else:
            # Default to jpg
            return ".jpg"

    def get_failed_urls(self) -> list[str]:
        """Get list of failed download URLs.

        Returns:
            List of URLs that failed to download
        """
        return self.failed_urls.copy()
