"""
Pinecone Tool for ARGUS.

Store data, perform semantic search, and rerank results.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class PineconeTool(BaseTool):
    """
    Pinecone - Vector database for semantic search.
    
    Features:
    - Store vectors with metadata
    - Semantic similarity search
    - Metadata filtering
    - Index management
    - Result reranking
    
    Example:
        >>> tool = PineconeTool(api_key="...", index_name="my-index")
        >>> result = tool(action="upsert", vectors=[...])
        >>> result = tool(action="query", vector=[...], top_k=5)
    """
    
    name = "pinecone"
    description = "Store data, perform semantic search, and rerank results"
    category = ToolCategory.DATA
    version = "1.0.0"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        
        self._pc = None
        self._index = None
        
        logger.debug(f"Pinecone initialized (index={index_name})")
    
    def _get_client(self):
        """Lazy-load Pinecone client."""
        if self._pc is None:
            try:
                from pinecone import Pinecone
                self._pc = Pinecone(api_key=self.api_key)
            except ImportError:
                raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")
        return self._pc
    
    def _get_index(self, index_name: Optional[str] = None):
        """Get index instance."""
        idx_name = index_name or self.index_name
        if not idx_name:
            raise ValueError("index_name is required")
        
        pc = self._get_client()
        return pc.Index(idx_name)
    
    def execute(
        self,
        action: str = "list_indexes",
        index_name: Optional[str] = None,
        vectors: Optional[list] = None,
        vector: Optional[list] = None,
        ids: Optional[list] = None,
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[dict] = None,
        include_values: bool = False,
        include_metadata: bool = True,
        dimension: int = 1536,
        metric: str = "cosine",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Pinecone operations."""
        actions = {
            "upsert": self._upsert,
            "query": self._query,
            "fetch": self._fetch,
            "delete": self._delete,
            "update": self._update,
            "list_indexes": self._list_indexes,
            "create_index": self._create_index,
            "delete_index": self._delete_index,
            "describe_index": self._describe_index,
            "describe_index_stats": self._describe_index_stats,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                index_name=index_name,
                vectors=vectors or [],
                vector=vector or [],
                ids=ids or [],
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                dimension=dimension,
                metric=metric,
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"Pinecone error: {e}")
            return ToolResult.from_error(f"Pinecone error: {e}")
    
    def _upsert(
        self,
        index_name: Optional[str] = None,
        vectors: Optional[list] = None,
        namespace: str = "",
        **kwargs,
    ) -> ToolResult:
        """Upsert vectors into the index."""
        if not vectors:
            return ToolResult.from_error("vectors is required")
        
        index = self._get_index(index_name)
        
        # Format vectors if needed
        formatted_vectors = []
        for v in vectors:
            if isinstance(v, dict):
                formatted_vectors.append(v)
            elif isinstance(v, (list, tuple)):
                # Assume (id, values, metadata) format
                if len(v) >= 2:
                    vec = {"id": str(v[0]), "values": v[1]}
                    if len(v) >= 3:
                        vec["metadata"] = v[2]
                    formatted_vectors.append(vec)
        
        # Generate IDs if not present
        for v in formatted_vectors:
            if "id" not in v:
                v["id"] = str(uuid.uuid4())[:12]
        
        response = index.upsert(vectors=formatted_vectors, namespace=namespace)
        
        return ToolResult.from_data({
            "upserted_count": response.get("upserted_count", len(formatted_vectors)),
            "namespace": namespace,
        })
    
    def _query(
        self,
        index_name: Optional[str] = None,
        vector: Optional[list] = None,
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[dict] = None,
        include_values: bool = False,
        include_metadata: bool = True,
        **kwargs,
    ) -> ToolResult:
        """Query the index for similar vectors."""
        if not vector:
            return ToolResult.from_error("vector is required")
        
        index = self._get_index(index_name)
        
        query_kwargs = {
            "vector": vector,
            "top_k": top_k,
            "namespace": namespace,
            "include_values": include_values,
            "include_metadata": include_metadata,
        }
        
        if filter:
            query_kwargs["filter"] = filter
        
        response = index.query(**query_kwargs)
        
        matches = []
        for match in response.get("matches", []):
            item = {
                "id": match["id"],
                "score": match["score"],
            }
            if include_metadata and "metadata" in match:
                item["metadata"] = match["metadata"]
            if include_values and "values" in match:
                item["values"] = match["values"]
            matches.append(item)
        
        return ToolResult.from_data({
            "matches": matches,
            "namespace": response.get("namespace", namespace),
        })
    
    def _fetch(
        self,
        index_name: Optional[str] = None,
        ids: Optional[list] = None,
        namespace: str = "",
        **kwargs,
    ) -> ToolResult:
        """Fetch vectors by ID."""
        if not ids:
            return ToolResult.from_error("ids is required")
        
        index = self._get_index(index_name)
        response = index.fetch(ids=ids, namespace=namespace)
        
        vectors = []
        for id_, data in response.get("vectors", {}).items():
            vectors.append({
                "id": id_,
                "values": data.get("values"),
                "metadata": data.get("metadata"),
            })
        
        return ToolResult.from_data({
            "vectors": vectors,
            "namespace": response.get("namespace", namespace),
        })
    
    def _delete(
        self,
        index_name: Optional[str] = None,
        ids: Optional[list] = None,
        namespace: str = "",
        delete_all: bool = False,
        filter: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete vectors from the index."""
        index = self._get_index(index_name)
        
        delete_kwargs = {"namespace": namespace}
        
        if delete_all:
            delete_kwargs["delete_all"] = True
        elif ids:
            delete_kwargs["ids"] = ids
        elif filter:
            delete_kwargs["filter"] = filter
        else:
            return ToolResult.from_error("ids, filter, or delete_all=True required")
        
        index.delete(**delete_kwargs)
        
        return ToolResult.from_data({
            "deleted": True,
            "namespace": namespace,
        })
    
    def _update(
        self,
        index_name: Optional[str] = None,
        id: Optional[str] = None,
        values: Optional[list] = None,
        set_metadata: Optional[dict] = None,
        namespace: str = "",
        **kwargs,
    ) -> ToolResult:
        """Update a vector."""
        if not id:
            return ToolResult.from_error("id is required")
        
        index = self._get_index(index_name)
        
        update_kwargs = {"id": id, "namespace": namespace}
        if values:
            update_kwargs["values"] = values
        if set_metadata:
            update_kwargs["set_metadata"] = set_metadata
        
        index.update(**update_kwargs)
        
        return ToolResult.from_data({
            "updated": True,
            "id": id,
        })
    
    def _list_indexes(self, **kwargs) -> ToolResult:
        """List all indexes."""
        pc = self._get_client()
        indexes = pc.list_indexes()
        
        index_list = []
        for idx in indexes:
            index_list.append({
                "name": idx.name,
                "dimension": idx.dimension,
                "metric": idx.metric,
                "host": idx.host,
            })
        
        return ToolResult.from_data({
            "indexes": index_list,
            "count": len(index_list),
        })
    
    def _create_index(
        self,
        index_name: Optional[str] = None,
        dimension: int = 1536,
        metric: str = "cosine",
        **kwargs,
    ) -> ToolResult:
        """Create a new index."""
        if not index_name:
            return ToolResult.from_error("index_name is required")
        
        from pinecone import ServerlessSpec
        
        pc = self._get_client()
        
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        return ToolResult.from_data({
            "index_name": index_name,
            "dimension": dimension,
            "metric": metric,
            "created": True,
        })
    
    def _delete_index(
        self,
        index_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete an index."""
        if not index_name:
            return ToolResult.from_error("index_name is required")
        
        pc = self._get_client()
        pc.delete_index(index_name)
        
        return ToolResult.from_data({
            "index_name": index_name,
            "deleted": True,
        })
    
    def _describe_index(
        self,
        index_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Describe an index."""
        idx_name = index_name or self.index_name
        if not idx_name:
            return ToolResult.from_error("index_name is required")
        
        pc = self._get_client()
        desc = pc.describe_index(idx_name)
        
        return ToolResult.from_data({
            "name": desc.name,
            "dimension": desc.dimension,
            "metric": desc.metric,
            "host": desc.host,
            "status": desc.status.state,
        })
    
    def _describe_index_stats(
        self,
        index_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get index statistics."""
        index = self._get_index(index_name)
        stats = index.describe_index_stats()
        
        return ToolResult.from_data({
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0),
            "namespaces": stats.get("namespaces", {}),
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["upsert", "query", "fetch", "delete", "update",
                                "list_indexes", "create_index", "delete_index",
                                "describe_index", "describe_index_stats"],
                    },
                    "index_name": {"type": "string"},
                    "vectors": {"type": "array"},
                    "vector": {"type": "array", "items": {"type": "number"}},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 10},
                    "namespace": {"type": "string", "default": ""},
                    "filter": {"type": "object"},
                    "dimension": {"type": "integer", "default": 1536},
                    "metric": {"type": "string", "default": "cosine"},
                },
                "required": ["action"],
            },
        }
