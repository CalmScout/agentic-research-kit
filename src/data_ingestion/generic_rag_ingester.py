"""Generic RAG ingester for configurable data schemas with GPU acceleration.

This ingester provides format-agnostic RAG ingestion with template-based content formatting,
making it suitable for any structured data source, not just claim verification datasets.
"""

import os
import io
import base64
import time
import asyncio
from io import BytesIO
import logging
from typing import Dict, Optional, Callable, Union, List
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# Import GPU models and singletons
from src.utils.vision_embedding import (
    Qwen3VisionEmbedder,
    Qwen3TextLLM,
    Qwen3VLTextLLM,
    Qwen3Embedding,
    Qwen3VLEmbedding,
    UnifiedQwen3VL,
    get_vision_model,
    get_text_llm,
    get_embedding_model,
    get_qwen2_llm,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Standalone Wrapper Functions for LightRAG (Avoids deepcopy OOM) ---
# Made ASYNC to satisfy LightRAG's internal execution model

async def hf_llm_wrapper(prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
    """Standalone ASYNC LLM wrapper that uses singletons to avoid capturing 'self'."""
    model_name = os.getenv("TEXT_LLM_MODEL", "Qwen/Qwen3-8B")
    device = os.getenv("DEVICE", "cuda")
    torch_dtype = os.getenv("TORCH_DTYPE", "float16")
    
    # Use singleton to get/load model
    llm = get_text_llm(model_name=model_name, device=device, torch_dtype=torch_dtype)
    return llm.generate(prompt, system_prompt=system_prompt)

async def hf_vision_wrapper(image_paths: List[str], **kwargs) -> List[str]:
    """Standalone ASYNC vision wrapper that uses singletons to avoid capturing 'self'."""
    model_name = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Thinking")
    device = os.getenv("DEVICE", "cuda")
    torch_dtype = os.getenv("TORCH_DTYPE", "float16")
    
    vision_model = get_vision_model(model_name=model_name, device=device, torch_dtype=torch_dtype)
    return vision_model.encode_images(image_paths)

async def hf_embedding_wrapper(texts: List[str], **kwargs) -> np.ndarray:
    """Standalone ASYNC embedding wrapper that uses singletons to avoid capturing 'self'."""
    model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    device = os.getenv("DEVICE", "cuda")
    torch_dtype = os.getenv("TORCH_DTYPE", "float16")
    
    embedding_model = get_embedding_model(model_name=model_name, device=device, torch_dtype=torch_dtype)
    
    # Handle both text and multimodal models
    if hasattr(embedding_model, 'embed_text_batch'):
        embeddings = embedding_model.embed_text_batch(texts)
    elif hasattr(embedding_model, 'embed_text'):
        embeddings = [embedding_model.embed_text(t) for t in texts]
        embeddings = np.array(embeddings)
    else:
        # Multimodal model - use text encoding
        embeddings = [embedding_model.encode(text) for text in texts]
        embeddings = np.array(embeddings)

    return embeddings

# --- End Wrapper Functions ---

class GenericRAGIngester:
    """Format-agnostic RAG ingester with template-based content formatting."""

    def __init__(
        self,
        working_dir: str = "./rag_storage",
        content_template: str = "{content}",
        content_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        use_gpu: bool = True,
        device: str = "cuda",
        torch_dtype: str = "float16"
    ):
        """Initialize generic RAG ingester with GPU support."""
        self.working_dir = working_dir
        # Fallback template if specific fields like 'title' are missing
        self.content_template = content_template
        self.content_fields = content_fields or ["content"]
        self.metadata_fields = metadata_fields or []

        # Check GPU availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = device if self.use_gpu else "cpu"
        self.torch_dtype = torch_dtype

        if self.use_gpu:
            logger.info(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("GPU not available, using CPU (slower)")

        # Load model names from environment or use defaults
        self.vision_model_name = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Thinking")
        self.text_llm_model_name = os.getenv("TEXT_LLM_MODEL", "Qwen/Qwen3-8B")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

        # Detect actual embedding dimension from model
        embedding_model = get_embedding_model(
            model_name=self.embedding_model_name,
            device=self.device,
            torch_dtype=self.torch_dtype
        )
        test_embedding = embedding_model.embed_text("test")
        actual_embedding_dim = test_embedding.shape[0]
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
        )

        logger.info(f"✓ LightRAG initialized (using async wrappers to prevent OOM)")

    async def ingest_df(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        images: Optional[Dict] = None
    ) -> Dict[str, any]:
        """Ingest DataFrame into RAG system using template-based formatting."""
        if images is None:
            images = {}

        # Initialize LightRAG storages
        await self.rag.initialize_storages()
        logger.info("✓ LightRAG storages initialized")

        content_list = []
        text_count = 0
        image_count = 0

        # Add text content
        for idx, row in df.iterrows():
            formatted_content = self._format_content(row)
            content_id = f"{id_column}_{idx}" if id_column else f"doc_{idx}"
            metadata = self._extract_metadata(row, idx, content_id)

            content_list.append({
                "content": formatted_content,
                "content_id": content_id,
                "metadata": metadata
            })
            text_count += 1

        # Add image content
        for doc_id, image_path in images.items():
            if Path(image_path).exists():
                if doc_id in df.index:
                    doc_row = df.loc[doc_id]
                    caption = self._format_content(doc_row, max_length=200)
                else:
                    caption = 'Document image'

                content_list.append({
                    "content": f"{caption}\nImage: {image_path}",
                    "content_id": f"image_{doc_id}",
                    "metadata": {
                        "type": "image",
                        "doc_id": doc_id,
                        "image_path": image_path,
                        "caption": caption
                    }
                })
                image_count += 1

        logger.info(f"Built content_list: {text_count} text items, {image_count} image items")

        try:
            batch_size = 10 # Reduced batch size for stability during intensive 8B processing
            total_items = len(content_list)
            start_time = time.time()

            for i in range(0, total_items, batch_size):
                batch = content_list[i:i+batch_size]
                # LightRAG handles its own internal parallelism/queuing via ainsert
                for item_idx, item in enumerate(batch):
                    item_start = time.time()
                    await self.rag.ainsert(item["content"])
                    item_time = time.time() - item_start
                    
                    current_item = i + item_idx + 1
                    logger.info(f"  [{current_item}/{total_items}] Ingested {item['metadata'].get('type', 'text')} ({item_time:.2f}s)")

            total_time = time.time() - start_time
            stats = {
                'text_items': text_count,
                'image_items': image_count,
                'total_items': text_count + image_count,
                'ingestion_time_seconds': total_time,
                'avg_time_per_item': total_time / total_items if total_items > 0 else 0
            }
            logger.info(f"✓ Ingestion complete: {stats['total_items']} items in {total_time:.1f}s")
            return stats

        except Exception as e:
            logger.error(f"Failed during ingestion: {e}")
            raise

    def _format_content(self, row: pd.Series, max_length: Optional[int] = None) -> str:
        """Format content using template and field values with safe fallback.
        
        Dynamically detects placeholders in content_template and replaces them 
        with corresponding values from the row.
        """
        import re
        
        # Find all {placeholder} patterns in the template
        placeholders = re.findall(r'\{([^{}]+)\}', self.content_template)
        
        formatted = self.content_template
        for p in placeholders:
            # Get value from row, default to empty string
            val = str(row.get(p, '')).strip() if p in row.index and not pd.isna(row.get(p)) else ''
            formatted = formatted.replace('{' + p + '}', val)

        # Fallback if no placeholders were replaced or result is too small
        if formatted.strip() == self.content_template.strip() or len(formatted.strip()) < 10:
            content_val = str(row.get('content', '')).strip()
            if content_val:
                formatted = content_val

        if max_length and len(formatted) > max_length:
            formatted = formatted[:max_length] + "..."
        return formatted

    def _extract_metadata(self, row: pd.Series, idx: int, content_id: str) -> Dict:
        """Extract metadata from data row."""
        metadata = {"type": "text", "doc_id": idx, "content_id": content_id}
        for field in self.metadata_fields:
            value = row.get(field, '')
            if not pd.isna(value):
                metadata[field] = str(value).strip()
        return metadata

    async def close(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ GPU cache cleared")
