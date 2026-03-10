"""Generic RAG ingester for configurable data schemas with GPU acceleration.

This ingester provides format-agnostic RAG ingestion with template-based content formatting,
making it suitable for any structured data source, not just claim verification datasets.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Standalone Wrapper Functions for LightRAG (Avoids deepcopy OOM) ---
# Made ASYNC to satisfy LightRAG's internal execution model


async def hf_llm_wrapper(prompt: str, system_prompt: str | None = None, **kwargs) -> str:
    """Standalone ASYNC LLM wrapper that uses vLLM API."""
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    model_name = os.getenv("TEXT_LLM_MODEL", "Qwen/Qwen3.5-4B")
    llm = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:8001/v1",
        api_key=SecretStr("EMPTY"),
        temperature=0.0,
        timeout=120.0,
    )

    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    res = await llm.ainvoke(messages)
    return str(res.content)


async def hf_vision_wrapper(image_paths: list[str], **kwargs) -> list[str]:
    """Standalone ASYNC vision wrapper. Note: Fallback implementation as we move to vLLM."""
    logger.warning("Vision wrapper called but not natively supported in vLLM text endpoint yet.")
    return ["Placeholder text for image"] * len(image_paths)


async def hf_embedding_wrapper(texts: list[str], **kwargs) -> np.ndarray:
    """Standalone ASYNC embedding wrapper that uses TEI API."""
    from langchain_openai import OpenAIEmbeddings
    from pydantic import SecretStr

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embeddings = OpenAIEmbeddings(
        model=model_name,
        base_url="http://localhost:8082/v1",
        api_key=SecretStr("EMPTY"),
        timeout=120.0,
    )

    res = await embeddings.aembed_documents(texts)
    return np.array(res, dtype=np.float32)


# --- End Wrapper Functions ---


class GenericRAGIngester:
    """Format-agnostic RAG ingester with template-based content formatting."""

    def __init__(
        self,
        working_dir: str = "./rag_storage",
        content_template: str = "{content}",
        content_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        use_gpu: bool = True,
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        """Initialize generic RAG ingester with API support."""
        self.working_dir = working_dir
        # Fallback template if specific fields like 'title' are missing
        self.content_template = content_template
        self.content_fields = content_fields or ["content"]
        self.metadata_fields = metadata_fields or []

        # Check GPU availability (just for logging now)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = device if self.use_gpu else "cpu"
        self.torch_dtype = torch_dtype

        if self.use_gpu:
            logger.info(
                f"✓ GPU acceleration info (not used directly, via API): {torch.cuda.get_device_name(0)}"
            )

        # Load model names from environment or use defaults
        self.vision_model_name = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Thinking")
        self.text_llm_model_name = os.getenv("TEXT_LLM_MODEL", "Qwen/Qwen3.5-4B")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")

        # Detect actual embedding dimension from API
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            base_url="http://localhost:8082/v1",
            api_key=SecretStr("EMPTY"),
        )
        test_embedding = embeddings.embed_query("test")
        actual_embedding_dim = len(test_embedding)
        logger.info(f"✓ Detected embedding dimension: {actual_embedding_dim}")

        # Setup LightRAG with async standalone functions
        self.embedding_func = EmbeddingFunc(
            embedding_dim=actual_embedding_dim,
            max_token_size=8192,
            func=hf_embedding_wrapper,
        )

        # Initialize LightRAG instance
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=hf_llm_wrapper,
            embedding_func=self.embedding_func,
            embedding_batch_num=1,  # Serial embedding to save RAM
            embedding_func_max_async=1,  # CRITICAL: Prevent OOM by using 1 worker locally
        )

        logger.info("✓ LightRAG initialized (using async API wrappers)")

    async def ingest_df(
        self, df: pd.DataFrame, id_column: str | None = None, images: dict | None = None
    ) -> dict[str, Any]:
        """Ingest DataFrame into RAG system using template-based formatting."""
        if images is None:
            images = {}

        # Initialize LightRAG storages
        await self.rag.initialize_storages()
        logger.info("✓ LightRAG storages initialized")

        content_list: list[dict[str, Any]] = []
        text_count = 0
        image_count = 0

        # Add text content
        for idx, row in df.iterrows():
            formatted_content = self._format_content(row)
            content_id = f"{id_column}_{idx}" if id_column else f"doc_{idx}"
            metadata = self._extract_metadata(row, idx, content_id)

            content_list.append(
                {"content": formatted_content, "content_id": content_id, "metadata": metadata}
            )
            text_count += 1

        # Add image content
        for doc_id, image_path in images.items():
            if Path(image_path).exists():
                if doc_id in df.index:
                    doc_row = df.loc[doc_id]
                    caption = self._format_content(doc_row, max_length=200)
                else:
                    caption = "Document image"

                content_list.append(
                    {
                        "content": f"{caption}\nImage: {image_path}",
                        "content_id": f"image_{doc_id}",
                        "metadata": {
                            "type": "image",
                            "doc_id": doc_id,
                            "image_path": image_path,
                            "caption": caption,
                        },
                    }
                )
                image_count += 1

        logger.info(f"Built content_list: {text_count} text items, {image_count} image items")

        try:
            batch_size = 10  # Reduced batch size for stability during intensive 8B processing
            total_items = len(content_list)
            start_time = time.time()

            for i in range(0, total_items, batch_size):
                batch = content_list[i : i + batch_size]
                # LightRAG handles its own internal parallelism/queuing via ainsert
                for item_idx, item in enumerate(batch):
                    item_start = time.time()
                    await self.rag.ainsert(item["content"])
                    item_time = time.time() - item_start

                    current_item = i + item_idx + 1
                    logger.info(
                        f"  [{current_item}/{total_items}] Ingested {item['metadata'].get('type', 'text')} ({item_time:.2f}s)"
                    )

            total_time = time.time() - start_time
            stats = {
                "text_items": text_count,
                "image_items": image_count,
                "total_items": text_count + image_count,
                "ingestion_time_seconds": total_time,
                "avg_time_per_item": total_time / total_items if total_items > 0 else 0,
            }
            logger.info(f"✓ Ingestion complete: {stats['total_items']} items in {total_time:.1f}s")
            return stats

        except Exception as e:
            logger.error(f"Failed during ingestion: {e}")
            raise

    def _format_content(self, row: pd.Series, max_length: int | None = None) -> str:
        """Format content using template and field values with safe fallback.

        Dynamically detects placeholders in content_template and replaces them
        with corresponding values from the row.
        """
        import re

        # Find all {placeholder} patterns in the template
        placeholders = re.findall(r"\{([^{}]+)\}", self.content_template)

        formatted = self.content_template
        for p in placeholders:
            # Get value from row, default to empty string
            val = str(row.get(p, "")).strip() if p in row.index and not pd.isna(row.get(p)) else ""
            formatted = formatted.replace("{" + p + "}", val)

        # Fallback if no placeholders were replaced or result is too small
        if formatted.strip() == self.content_template.strip() or len(formatted.strip()) < 10:
            content_val = str(row.get("content", "")).strip()
            if content_val:
                formatted = content_val

        if max_length and len(formatted) > max_length:
            formatted = formatted[:max_length] + "..."
        return formatted

    def _extract_metadata(self, row: pd.Series, idx: int, content_id: str) -> dict:
        """Extract metadata from data row."""
        metadata = {"type": "text", "doc_id": idx, "content_id": content_id}
        for field in self.metadata_fields:
            value = row.get(field, "")
            if not pd.isna(value):
                metadata[field] = str(value).strip()
        return metadata

    async def close(self):
        """Clean up resources."""
        # API doesn't require local cleanup
        pass
