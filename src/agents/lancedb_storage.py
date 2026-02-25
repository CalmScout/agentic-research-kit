import os
import json
import asyncio
from typing import Any, Union, final, List
from dataclasses import dataclass, field

import numpy as np
import lancedb
import pyarrow as pa
from loguru import logger
from lightrag.base import (
    BaseKVStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    BaseVectorStorage,
)
from lightrag.exceptions import StorageNotInitializedError
from lightrag.kg.shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
)
from lightrag.utils import get_pinyin_sort_key, compute_mdhash_id


@final
@dataclass
class LanceDBKVStorage(BaseKVStorage):
    """LanceDB implementation for Key-Value storage."""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        # Use a single lancedb database directory for the workspace
        self.db_path = os.path.join(workspace_dir, "lancedb_store")
        self.table_name = f"kv_{self.namespace}"
        
        self._db = None
        self._table = None
        self._storage_lock = None
        self.storage_updated = None

    async def initialize(self):
        """Initialize LanceDB connection and table."""
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        
        async with self._storage_lock:
            self._db = lancedb.connect(self.db_path)
            try:
                self._table = self._db.open_table(self.table_name)
                logger.info(f"[{self.workspace}] Opened existing LanceDB table {self.table_name}")
            except Exception:
                # Table doesn't exist, we will create it on first upsert
                self._table = None
                logger.info(f"[{self.workspace}] LanceDB table {self.table_name} will be created on first use")

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBKVStorage")
        
        async with self._storage_lock:
            if self._table is None:
                return None
            try:
                results = self._table.search().where(f"id = '{id}'").to_pandas()
                if not results.empty:
                    data_str = results.iloc[0]["data"]
                    result = json.loads(data_str)
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    result["_id"] = id
                    return result
            except Exception as e:
                logger.error(f"Error querying LanceDB for id {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBKVStorage")
            
        async with self._storage_lock:
            if self._table is None:
                return [None for _ in ids]
                
            results = []
            try:
                # Use pandas filtering for multiple IDs
                df = self._table.search().to_pandas()
                df_filtered = df[df["id"].isin(ids)]
                
                lookup = {}
                for _, row in df_filtered.iterrows():
                    data_str = row["data"]
                    parsed = json.loads(data_str)
                    parsed.setdefault("create_time", 0)
                    parsed.setdefault("update_time", 0)
                    parsed["_id"] = row["id"]
                    lookup[row["id"]] = parsed
                    
                for id in ids:
                    results.append(lookup.get(id))
            except Exception as e:
                logger.error(f"Error querying LanceDB for multiple ids: {e}")
                results = [None for _ in ids]
            return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBKVStorage")
            
        async with self._storage_lock:
            if self._table is None:
                return keys
            try:
                df = self._table.search().to_pandas()
                existing_keys = set(df["id"].tolist())
                return keys - existing_keys
            except Exception as e:
                logger.error(f"Error filtering keys in LanceDB: {e}")
                return keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
            
        import time
        current_time = int(time.time())

        logger.debug(f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}")
        
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBKVStorage")
            
        async with self._storage_lock:
            upsert_data = []
            ids_to_update = list(data.keys())
            
            if self._table is not None:
                try:
                    # In LanceDB, we delete existing rows before adding updated ones
                    in_clause = ", ".join([repr(i) for i in ids_to_update])
                    self._table.delete(f"id IN ({in_clause})")
                except Exception:
                    pass

            for k, v in data.items():
                if self.namespace.endswith("text_chunks") and "llm_cache_list" not in v:
                    v["llm_cache_list"] = []

                v["update_time"] = current_time
                v.setdefault("create_time", current_time)
                v["_id"] = k
                
                upsert_data.append({
                    "id": k,
                    "data": json.dumps(v)
                })

            if self._table is None:
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("data", pa.string())
                ])
                self._table = self._db.create_table(self.table_name, data=upsert_data, schema=schema)
                logger.info(f"[{self.workspace}] Created LanceDB table {self.table_name}")
            else:
                self._table.add(upsert_data)

            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def delete(self, ids: list[str]) -> None:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBKVStorage")
            
        async with self._storage_lock:
            if self._table is None:
                return
            try:
                in_clause = ", ".join([repr(i) for i in ids])
                self._table.delete(f"id IN ({in_clause})")
                await set_all_update_flags(self.namespace, workspace=self.workspace)
            except Exception as e:
                logger.error(f"Error deleting from LanceDB: {e}")

    async def is_empty(self) -> bool:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBKVStorage")
            
        async with self._storage_lock:
            if self._table is None:
                return True
            try:
                return self._table.count_rows() == 0
            except Exception:
                return True

    async def drop(self) -> dict[str, str]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBKVStorage")
            
        try:
            async with self._storage_lock:
                if self._table is not None:
                    self._db.drop_table(self.table_name)
                    self._table = None
                    await set_all_update_flags(self.namespace, workspace=self.workspace)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def finalize(self):
        pass


@final
@dataclass
class LanceDBDocStatusStorage(DocStatusStorage):
    """LanceDB implementation for Document Status storage."""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self.db_path = os.path.join(workspace_dir, "lancedb_store")
        self.table_name = f"doc_status_{self.namespace}"
        
        self._db = None
        self._table = None
        self._storage_lock = None
        self.storage_updated = None

    async def initialize(self):
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        
        async with self._storage_lock:
            self._db = lancedb.connect(self.db_path)
            try:
                self._table = self._db.open_table(self.table_name)
            except Exception:
                self._table = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBDocStatusStorage")
        async with self._storage_lock:
            if self._table is None:
                return keys
            try:
                df = self._table.search().to_pandas()
                existing_keys = set(df["id"].tolist())
                return keys - existing_keys
            except Exception:
                return keys

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        ordered_results = []
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBDocStatusStorage")
            
        async with self._storage_lock:
            if self._table is None:
                return [None for _ in ids]
                
            try:
                df = self._table.search().to_pandas()
                df_filtered = df[df["id"].isin(ids)]
                
                lookup = {}
                for _, row in df_filtered.iterrows():
                    lookup[row["id"]] = json.loads(row["data"])
                    
                for id in ids:
                    ordered_results.append(lookup.get(id))
            except Exception:
                ordered_results = [None for _ in ids]
        return ordered_results

    async def get_status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in DocStatus}
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBDocStatusStorage")
            
        async with self._storage_lock:
            if self._table is not None:
                try:
                    df = self._table.search().to_pandas()
                    for data_str in df["data"]:
                        doc = json.loads(data_str)
                        status = doc.get("status")
                        if status in counts:
                            counts[status] += 1
                except Exception as e:
                    logger.error(f"Error getting status counts: {e}")
        return counts

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        result = {}
        async with self._storage_lock:
            if self._table is not None:
                try:
                    df = self._table.search().to_pandas()
                    for _, row in df.iterrows():
                        doc = json.loads(row["data"])
                        if doc.get("status") == status.value:
                            doc.pop("content", None)
                            if "file_path" not in doc:
                                doc["file_path"] = "no-file-path"
                            if "metadata" not in doc:
                                doc["metadata"] = {}
                            if "error_msg" not in doc:
                                doc["error_msg"] = None
                            result[row["id"]] = DocProcessingStatus(**doc)
                except Exception as e:
                    logger.error(f"Error getting docs by status: {e}")
        return result

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        result = {}
        async with self._storage_lock:
            if self._table is not None:
                try:
                    df = self._table.search().to_pandas()
                    for _, row in df.iterrows():
                        doc = json.loads(row["data"])
                        if doc.get("track_id") == track_id:
                            doc.pop("content", None)
                            if "file_path" not in doc:
                                doc["file_path"] = "no-file-path"
                            if "metadata" not in doc:
                                doc["metadata"] = {}
                            if "error_msg" not in doc:
                                doc["error_msg"] = None
                            result[row["id"]] = DocProcessingStatus(**doc)
                except Exception as e:
                    logger.error(f"Error getting docs by track id: {e}")
        return result

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            if self._table is None:
                return None
            try:
                df = self._table.search().to_pandas()
                for _, row in df.iterrows():
                    doc = json.loads(row["data"])
                    if doc.get("file_path") == file_path:
                        doc["_id"] = row["id"]
                        return doc
            except Exception as e:
                logger.error(f"Error getting doc by file path: {e}")
        return None

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
            
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBDocStatusStorage")
            
        async with self._storage_lock:
            upsert_data = []
            ids_to_update = list(data.keys())
            
            if self._table is not None:
                try:
                    in_clause = ", ".join([repr(i) for i in ids_to_update])
                    self._table.delete(f"id IN ({in_clause})")
                except Exception:
                    pass

            for doc_id, doc_data in data.items():
                if "chunks_list" not in doc_data:
                    doc_data["chunks_list"] = []
                
                upsert_data.append({
                    "id": doc_id,
                    "data": json.dumps(doc_data)
                })

            if self._table is None:
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("data", pa.string())
                ])
                self._table = self._db.create_table(self.table_name, data=upsert_data, schema=schema)
            else:
                self._table.add(upsert_data)

            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def delete(self, ids: list[str]) -> None:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBDocStatusStorage")
            
        async with self._storage_lock:
            if self._table is None:
                return
            try:
                in_clause = ", ".join([repr(i) for i in ids])
                self._table.delete(f"id IN ({in_clause})")
                await set_all_update_flags(self.namespace, workspace=self.workspace)
            except Exception as e:
                logger.error(f"Error deleting from LanceDB: {e}")

    async def is_empty(self) -> bool:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBDocStatusStorage")
        async with self._storage_lock:
            if self._table is None:
                return True
            try:
                return self._table.count_rows() == 0
            except Exception:
                return True

    async def drop(self) -> dict[str, str]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBDocStatusStorage")
            
        try:
            async with self._storage_lock:
                if self._table is not None:
                    self._db.drop_table(self.table_name)
                    self._table = None
                    await set_all_update_flags(self.namespace, workspace=self.workspace)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._storage_lock:
            if self._table is None:
                return None
            try:
                results = self._table.search().where(f"id = '{id}'").to_pandas()
                if not results.empty:
                    return json.loads(results.iloc[0]["data"])
            except Exception:
                pass
            return None

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"
        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        all_docs = []
        async with self._storage_lock:
            if self._table is None:
                return [], 0
                
            try:
                df = self._table.search().to_pandas()
                for _, row in df.iterrows():
                    doc_id = row["id"]
                    doc_data = json.loads(row["data"])
                    
                    if status_filter is not None and doc_data.get("status") != status_filter.value:
                        continue
                        
                    doc_data.pop("content", None)
                    if "file_path" not in doc_data:
                        doc_data["file_path"] = "no-file-path"
                    if "metadata" not in doc_data:
                        doc_data["metadata"] = {}
                    if "error_msg" not in doc_data:
                        doc_data["error_msg"] = None

                    doc_status = DocProcessingStatus(**doc_data)

                    if sort_field == "id":
                        doc_status._sort_key = doc_id
                    elif sort_field == "file_path":
                        file_path_value = getattr(doc_status, sort_field, "")
                        doc_status._sort_key = get_pinyin_sort_key(file_path_value)
                    else:
                        doc_status._sort_key = getattr(doc_status, sort_field, "")

                    all_docs.append((doc_id, doc_status))
            except Exception as e:
                logger.error(f"Error in paginated docs: {e}")

        reverse = sort_direction.lower() == "desc"
        all_docs.sort(key=lambda x: x[1]._sort_key, reverse=reverse)

        total_count = len(all_docs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return all_docs[start_idx:end_idx], total_count


@final
@dataclass
class LanceDBVectorDBStorage(BaseVectorStorage):
    """LanceDB implementation for Vector storage."""

    def __post_init__(self):
        self._validate_embedding_func()
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self.db_path = os.path.join(workspace_dir, "lancedb_store")
        self.table_name = f"vdb_{self.namespace}"
        
        self._db = None
        self._table = None
        self._storage_lock = None
        self.storage_updated = None
        
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

    async def initialize(self):
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        
        async with self._storage_lock:
            self._db = lancedb.connect(self.db_path)
            try:
                self._table = self._db.open_table(self.table_name)
            except Exception:
                self._table = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        import time
        current_time = int(time.time())
        
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        upsert_data = []
        for i, (k, v) in enumerate(data.items()):
            metadata = {k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            metadata["__created_at__"] = current_time
            
            upsert_data.append({
                "id": k,
                "vector": embeddings[i].tolist(),
                "metadata": json.dumps(metadata)
            })

        if self._storage_lock is None:
            raise StorageNotInitializedError("LanceDBVectorDBStorage")
            
        async with self._storage_lock:
            if self._table is not None:
                try:
                    in_clause = ", ".join([repr(i) for i in data.keys()])
                    self._table.delete(f"id IN ({in_clause})")
                except Exception:
                    pass

            if self._table is None:
                dim = self.embedding_func.embedding_dim
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("metadata", pa.string())
                ])
                self._table = self._db.create_table(self.table_name, data=upsert_data, schema=schema)
            else:
                self._table.add(upsert_data)

            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = await self.embedding_func([query], _priority=5)
            embedding = embedding[0]

        async with self._storage_lock:
            if self._table is None:
                return []
            
            results = self._table.search(embedding).distance_type("cosine").limit(top_k).to_pandas()
            
            formatted_results = []
            for _, row in results.iterrows():
                # LanceDB cosine distance is 1 - similarity. 
                similarity = 1.0 - row["_distance"]
                if similarity < self.cosine_better_than_threshold:
                    continue
                    
                meta = json.loads(row["metadata"])
                res = {
                    **meta,
                    "id": row["id"],
                    "distance": similarity,
                    "created_at": meta.get("__created_at__")
                }
                formatted_results.append(res)
            return formatted_results

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            if self._table is None:
                return None
            results = self._table.search().where(f"id = '{id}'").to_pandas()
            if not results.empty:
                row = results.iloc[0]
                meta = json.loads(row["metadata"])
                return {**meta, "id": row["id"], "vector": row["vector"]}
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            if self._table is None:
                return []
            df = self._table.search().to_pandas()
            df_filtered = df[df["id"].isin(ids)]
            results = []
            for _, row in df_filtered.iterrows():
                meta = json.loads(row["metadata"])
                results.append({**meta, "id": row["id"], "vector": row["vector"]})
            return results

    async def delete(self, ids: list[str]):
        async with self._storage_lock:
            if self._table is None:
                return
            in_clause = ", ".join([repr(i) for i in ids])
            self._table.delete(f"id IN ({in_clause})")
            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        async with self._storage_lock:
            if self._table is None:
                return
            df = self._table.search().to_pandas()
            ids_to_delete = []
            for _, row in df.iterrows():
                meta = json.loads(row["metadata"])
                if meta.get("src_id") == entity_name or meta.get("tgt_id") == entity_name:
                    ids_to_delete.append(row["id"])
            
            if ids_to_delete:
                in_clause = ", ".join([repr(i) for i in ids_to_delete])
                self._table.delete(f"id IN ({in_clause})")
                await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        async with self._storage_lock:
            if self._table is None:
                return {}
            df = self._table.search().to_pandas()
            df_filtered = df[df["id"].isin(ids)]
            return {row["id"]: row["vector"].tolist() for _, row in df_filtered.iterrows()}

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def is_empty(self) -> bool:
        async with self._storage_lock:
            if self._table is None:
                return True
            return self._table.count_rows() == 0

    async def drop(self) -> dict[str, str]:
        async with self._storage_lock:
            if self._table is not None:
                self._db.drop_table(self.table_name)
                self._table = None
                await set_all_update_flags(self.namespace, workspace=self.workspace)
        return {"status": "success", "message": "data dropped"}
