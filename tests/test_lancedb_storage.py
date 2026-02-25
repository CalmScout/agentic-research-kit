import asyncio
import logging
import os
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_lancedb_storage():
    """Test LanceDBKVStorage implementation."""
    from src.agents.lancedb_storage import LanceDBKVStorage
    from lightrag.kg.shared_storage import (
        initialize_share_data, 
        try_initialize_namespace,
        get_data_init_lock
    )
    
    # Test directory
    test_dir = "./test_rag_storage"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    global_config = {"working_dir": test_dir}
    
    try:
        # Initialize LightRAG shared data structures
        initialize_share_data()
        
        namespace = "test_namespace"
        workspace = "test_workspace"
        
        async with get_data_init_lock():
            await try_initialize_namespace(namespace, workspace=workspace)

        # Mock embedding function
        from unittest.mock import MagicMock
        mock_embedding = MagicMock()
        mock_embedding.embedding_dim = 2048

        # Initialize storage
        storage = LanceDBKVStorage(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=mock_embedding
        )
        
        await storage.initialize()
        logger.info("✓ Storage initialized")
        
        # Test is_empty
        is_empty = await storage.is_empty()
        assert is_empty is True
        logger.info("✓ is_empty works for empty storage")
        
        # Test upsert
        test_data = {
            "key1": {"content": "value1", "meta": "info1"},
            "key2": {"content": "value2", "meta": "info2"}
        }
        await storage.upsert(test_data)
        logger.info("✓ upsert works")
        
        # Test get_by_id
        res1 = await storage.get_by_id("key1")
        assert res1 is not None
        assert res1["content"] == "value1"
        assert res1["_id"] == "key1"
        logger.info("✓ get_by_id works")
        
        # Test get_by_ids
        res_list = await storage.get_by_ids(["key1", "key2", "nonexistent"])
        assert len(res_list) == 3
        assert res_list[0]["content"] == "value1"
        assert res_list[1]["content"] == "value2"
        assert res_list[2] is None
        logger.info("✓ get_by_ids works")
        
        # Test filter_keys
        keys_to_filter = {"key1", "key3"}
        remaining_keys = await storage.filter_keys(keys_to_filter)
        assert remaining_keys == {"key3"}
        logger.info("✓ filter_keys works")
        
        # Test delete
        await storage.delete(["key1"])
        res1_after = await storage.get_by_id("key1")
        assert res1_after is None
        logger.info("✓ delete works")
        
        # Test drop
        await storage.drop()
        is_empty_after = await storage.is_empty()
        assert is_empty_after is True
        logger.info("✓ drop works")
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

async def test_lancedb_doc_status_storage():
    """Test LanceDBDocStatusStorage implementation."""
    from src.agents.lancedb_storage import LanceDBDocStatusStorage
    from lightrag.base import DocStatus, DocProcessingStatus
    from lightrag.kg.shared_storage import (
        initialize_share_data, 
        try_initialize_namespace,
        get_data_init_lock
    )
    
    test_dir = "./test_rag_storage_doc"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    global_config = {"working_dir": test_dir}
    
    try:
        initialize_share_data()
        namespace = "test_doc_namespace"
        workspace = "test_doc_workspace"
        
        async with get_data_init_lock():
            await try_initialize_namespace(namespace, workspace=workspace)

        # Mock embedding function
        from unittest.mock import MagicMock
        mock_embedding = MagicMock()

        storage = LanceDBDocStatusStorage(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=mock_embedding
        )
        
        await storage.initialize()
        logger.info("✓ DocStatus storage initialized")
        
        # Test upsert
        from datetime import datetime, timedelta
        now = datetime.now()
        doc1_time = now.isoformat()
        doc2_time = (now + timedelta(minutes=1)).isoformat()
        
        doc1 = {
            "status": DocStatus.PROCESSED.value,
            "file_path": "path1.txt",
            "content_summary": "Summary 1",
            "content_length": 100,
            "updated_at": doc1_time,
            "created_at": doc1_time
        }
        doc2 = {
            "status": DocStatus.PENDING.value,
            "file_path": "path2.txt",
            "content_summary": "Summary 2",
            "content_length": 200,
            "updated_at": doc2_time,
            "created_at": doc2_time
        }
        await storage.upsert({"doc1": doc1, "doc2": doc2})
        logger.info("✓ DocStatus upsert works")
        
        # Test get_status_counts
        counts = await storage.get_status_counts()
        assert counts[DocStatus.PROCESSED.value] == 1
        assert counts[DocStatus.PENDING.value] == 1
        logger.info("✓ get_status_counts works")
        
        # Test get_docs_by_status
        processed_docs = await storage.get_docs_by_status(DocStatus.PROCESSED)
        assert len(processed_docs) == 1
        assert "doc1" in processed_docs
        assert isinstance(processed_docs["doc1"], DocProcessingStatus)
        logger.info("✓ get_docs_by_status works")
        
        # Test get_docs_paginated
        docs, total = await storage.get_docs_paginated(page=1, page_size=10, sort_field="updated_at", sort_direction="desc")
        assert total == 2
        assert docs[0][0] == "doc2" # Higher updated_at
        assert docs[1][0] == "doc1"
        logger.info("✓ get_docs_paginated works")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

async def test_lancedb_vector_db_storage():
    """Test LanceDBVectorDBStorage implementation."""
    from src.agents.lancedb_storage import LanceDBVectorDBStorage
    import numpy as np
    from lightrag.kg.shared_storage import (
        initialize_share_data, 
        try_initialize_namespace,
        get_data_init_lock
    )
    
    test_dir = "./test_rag_storage_vector"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    global_config = {
        "working_dir": test_dir,
        "embedding_batch_num": 32
    }
    
    try:
        initialize_share_data()
        namespace = "test_vector_namespace"
        workspace = "test_vector_workspace"
        
        async with get_data_init_lock():
            await try_initialize_namespace(namespace, workspace=workspace)

        # Mock embedding function
        async def mock_emb_func(texts, **kwargs):
            return [np.random.rand(128).astype(np.float32) for _ in texts]
            
        from unittest.mock import MagicMock
        mock_embedding = MagicMock(side_effect=mock_emb_func)
        mock_embedding.embedding_dim = 128

        storage = LanceDBVectorDBStorage(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=mock_embedding,
            meta_fields={"content", "file_path"}
        )
        
        await storage.initialize()
        logger.info("✓ VectorDB storage initialized")
        
        # Test upsert
        test_data = {
            "vec1": {"content": "content 1", "file_path": "file1.txt"},
            "vec2": {"content": "content 2", "file_path": "file2.txt"}
        }
        await storage.upsert(test_data)
        logger.info("✓ VectorDB upsert works")
        
        # Test query
        # We need a 128-dim query embedding
        query_emb = np.random.rand(128).astype(np.float32).tolist()
        results = await storage.query("test query", top_k=5, query_embedding=query_emb)
        assert len(results) == 2
        assert "content" in results[0]
        assert "file_path" in results[0]
        logger.info("✓ VectorDB query works")
        
        # Test get_by_id
        res1 = await storage.get_by_id("vec1")
        assert res1 is not None
        assert res1["id"] == "vec1"
        assert "vector" in res1
        logger.info("✓ VectorDB get_by_id works")
        
        # Test delete
        await storage.delete(["vec1"])
        res1_after = await storage.get_by_id("vec1")
        assert res1_after is None
        logger.info("✓ VectorDB delete works")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    async def main():
        await test_lancedb_storage()
        await test_lancedb_doc_status_storage()
        await test_lancedb_vector_db_storage()
        
    asyncio.run(main())
