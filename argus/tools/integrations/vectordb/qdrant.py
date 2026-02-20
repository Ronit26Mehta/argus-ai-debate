"""
Qdrant Tool for ARGUS.

Store and retrieve information using semantic vector search with Qdrant.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class QdrantTool(BaseTool):
    """
    Qdrant - Vector search engine.
    
    Features:
    - Store vectors with payload
    - Semantic similarity search
    - Filtering and scoring
    - Collection management
    - Batch operations
    
    Example:
        >>> tool = QdrantTool(host="localhost", port=6333)
        >>> result = tool(action="upsert", collection="docs", vectors=[...])
        >>> result = tool(action="search", collection="docs", query_vector=[...])
    """
    
    name = "qdrant"
    description = "Store and retrieve information using semantic vector search"
    category = ToolCategory.DATA
    version = "1.0.0"
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 6333,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.url = url or os.getenv("QDRANT_URL")
        
        self._client = None
        
        logger.debug(f"Qdrant initialized (host={self.host}:{self.port})")
    
    def _get_client(self):
        """Lazy-load Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                
                if self.url:
                    self._client = QdrantClient(url=self.url, api_key=self.api_key)
                else:
                    self._client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
            except ImportError:
                raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
        return self._client
    
    def execute(
        self,
        action: str = "list_collections",
        collection: Optional[str] = None,
        vectors: Optional[list] = None,
        query_vector: Optional[list] = None,
        ids: Optional[list] = None,
        limit: int = 10,
        filter: Optional[dict] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        dimension: int = 1536,
        distance: str = "Cosine",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Qdrant operations."""
        actions = {
            "upsert": self._upsert,
            "search": self._search,
            "retrieve": self._retrieve,
            "delete": self._delete,
            "scroll": self._scroll,
            "create_collection": self._create_collection,
            "delete_collection": self._delete_collection,
            "list_collections": self._list_collections,
            "get_collection": self._get_collection,
            "count": self._count,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                collection=collection,
                vectors=vectors or [],
                query_vector=query_vector or [],
                ids=ids or [],
                limit=limit,
                filter=filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                dimension=dimension,
                distance=distance,
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"Qdrant error: {e}")
            return ToolResult.from_error(f"Qdrant error: {e}")
    
    def _upsert(
        self,
        collection: Optional[str] = None,
        vectors: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Upsert points into a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not vectors:
            return ToolResult.from_error("vectors is required")
        
        from qdrant_client.models import PointStruct
        
        client = self._get_client()
        
        points = []
        for i, v in enumerate(vectors):
            if isinstance(v, dict):
                point_id = v.get("id", str(uuid.uuid4())[:12])
                vector = v.get("vector", v.get("values", []))
                payload = v.get("payload", v.get("metadata", {}))
            elif isinstance(v, (list, tuple)):
                point_id = str(uuid.uuid4())[:12]
                vector = v
                payload = {}
            else:
                continue
            
            points.append(PointStruct(
                id=point_id if isinstance(point_id, int) else hash(point_id) & 0x7FFFFFFF,
                vector=vector,
                payload=payload,
            ))
        
        client.upsert(collection_name=collection, points=points)
        
        return ToolResult.from_data({
            "collection": collection,
            "upserted": len(points),
        })
    
    def _search(
        self,
        collection: Optional[str] = None,
        query_vector: Optional[list] = None,
        limit: int = 10,
        filter: Optional[dict] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> ToolResult:
        """Search for similar vectors."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not query_vector:
            return ToolResult.from_error("query_vector is required")
        
        client = self._get_client()
        
        search_kwargs = {
            "collection_name": collection,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": with_vectors,
        }
        
        if filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            # Simple filter conversion
            conditions = []
            for key, value in filter.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            search_kwargs["query_filter"] = Filter(must=conditions)
        
        results = client.search(**search_kwargs)
        
        matches = []
        for hit in results:
            item = {
                "id": hit.id,
                "score": hit.score,
            }
            if with_payload:
                item["payload"] = hit.payload
            if with_vectors:
                item["vector"] = hit.vector
            matches.append(item)
        
        return ToolResult.from_data({
            "collection": collection,
            "results": matches,
            "count": len(matches),
        })
    
    def _retrieve(
        self,
        collection: Optional[str] = None,
        ids: Optional[list] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> ToolResult:
        """Retrieve points by ID."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not ids:
            return ToolResult.from_error("ids is required")
        
        client = self._get_client()
        
        # Convert string IDs to hashes if needed
        point_ids = []
        for id_ in ids:
            if isinstance(id_, int):
                point_ids.append(id_)
            else:
                point_ids.append(hash(id_) & 0x7FFFFFFF)
        
        results = client.retrieve(
            collection_name=collection,
            ids=point_ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        
        points = []
        for point in results:
            item = {"id": point.id}
            if with_payload:
                item["payload"] = point.payload
            if with_vectors:
                item["vector"] = point.vector
            points.append(item)
        
        return ToolResult.from_data({
            "collection": collection,
            "points": points,
            "count": len(points),
        })
    
    def _delete(
        self,
        collection: Optional[str] = None,
        ids: Optional[list] = None,
        filter: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete points from a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        
        from qdrant_client.models import PointIdsList, FilterSelector, Filter, FieldCondition, MatchValue
        
        if ids:
            point_ids = []
            for id_ in ids:
                if isinstance(id_, int):
                    point_ids.append(id_)
                else:
                    point_ids.append(hash(id_) & 0x7FFFFFFF)
            
            client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=point_ids),
            )
        elif filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            
            client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=Filter(must=conditions)),
            )
        else:
            return ToolResult.from_error("ids or filter required")
        
        return ToolResult.from_data({
            "collection": collection,
            "deleted": True,
        })
    
    def _scroll(
        self,
        collection: Optional[str] = None,
        limit: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False,
        offset: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Scroll through all points."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            offset=offset,
        )
        
        points = []
        for point in results:
            item = {"id": point.id}
            if with_payload:
                item["payload"] = point.payload
            if with_vectors:
                item["vector"] = point.vector
            points.append(item)
        
        return ToolResult.from_data({
            "collection": collection,
            "points": points,
            "count": len(points),
            "next_offset": next_offset,
        })
    
    def _create_collection(
        self,
        collection: Optional[str] = None,
        dimension: int = 1536,
        distance: str = "Cosine",
        **kwargs,
    ) -> ToolResult:
        """Create a new collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        from qdrant_client.models import Distance, VectorParams
        
        client = self._get_client()
        
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=distance_map.get(distance.lower(), Distance.COSINE),
            ),
        )
        
        return ToolResult.from_data({
            "collection": collection,
            "dimension": dimension,
            "distance": distance,
            "created": True,
        })
    
    def _delete_collection(
        self,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        client.delete_collection(collection_name=collection)
        
        return ToolResult.from_data({
            "collection": collection,
            "deleted": True,
        })
    
    def _list_collections(self, **kwargs) -> ToolResult:
        """List all collections."""
        client = self._get_client()
        collections = client.get_collections()
        
        return ToolResult.from_data({
            "collections": [
                {"name": c.name}
                for c in collections.collections
            ],
            "count": len(collections.collections),
        })
    
    def _get_collection(
        self,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get collection info."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        info = client.get_collection(collection_name=collection)
        
        return ToolResult.from_data({
            "name": collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        })
    
    def _count(
        self,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Count points in a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        count = client.count(collection_name=collection)
        
        return ToolResult.from_data({
            "collection": collection,
            "count": count.count,
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["upsert", "search", "retrieve", "delete", "scroll",
                                "create_collection", "delete_collection", "list_collections",
                                "get_collection", "count"],
                    },
                    "collection": {"type": "string"},
                    "vectors": {"type": "array"},
                    "query_vector": {"type": "array", "items": {"type": "number"}},
                    "ids": {"type": "array"},
                    "limit": {"type": "integer", "default": 10},
                    "filter": {"type": "object"},
                    "dimension": {"type": "integer", "default": 1536},
                    "distance": {"type": "string", "default": "Cosine"},
                },
                "required": ["action"],
            },
        }
