"""
Chroma Tool for ARGUS.

Store and retrieve information using semantic vector search with Chroma.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Optional, Any
from pathlib import Path

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class ChromaTool(BaseTool):
    """
    Chroma - Open-source embedding database.
    
    Features:
    - Store documents with embeddings
    - Semantic similarity search
    - Metadata filtering
    - Collection management
    - Persistent and in-memory storage
    
    Example:
        >>> tool = ChromaTool(persist_directory="./chroma_db")
        >>> result = tool(action="add", collection="docs", documents=["Hello world"])
        >>> result = tool(action="query", collection="docs", query_text="greeting")
    """
    
    name = "chroma"
    description = "Store and retrieve information using semantic vector search"
    category = ToolCategory.DATA
    version = "1.0.0"
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8000,
        embedding_function: Optional[str] = "default",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.persist_directory = persist_directory
        self.host = host or os.getenv("CHROMA_HOST")
        self.port = port
        self.embedding_function = embedding_function
        
        self._client = None
        self._embedder = None
        
        logger.debug(f"Chroma initialized (persist={persist_directory}, host={host})")
    
    def _get_client(self):
        """Lazy-load Chroma client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                if self.host:
                    # Connect to remote server
                    self._client = chromadb.HttpClient(host=self.host, port=self.port)
                elif self.persist_directory:
                    # Persistent local client
                    self._client = chromadb.PersistentClient(path=self.persist_directory)
                else:
                    # In-memory client
                    self._client = chromadb.Client()
                    
            except ImportError:
                raise ImportError("chromadb not installed. Run: pip install chromadb")
        return self._client
    
    def _get_embedding_function(self):
        """Get embedding function."""
        if self._embedder is None:
            try:
                import chromadb.utils.embedding_functions as embedding_functions
                
                if self.embedding_function == "default":
                    self._embedder = embedding_functions.DefaultEmbeddingFunction()
                elif self.embedding_function == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                    self._embedder = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key,
                        model_name="text-embedding-ada-002"
                    )
                elif self.embedding_function == "sentence_transformers":
                    self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction()
                else:
                    self._embedder = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                self._embedder = None
        return self._embedder
    
    def execute(
        self,
        action: str = "list_collections",
        collection: Optional[str] = None,
        documents: Optional[list] = None,
        embeddings: Optional[list] = None,
        metadatas: Optional[list] = None,
        ids: Optional[list] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[list] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Chroma operations."""
        actions = {
            "add": self._add,
            "query": self._query,
            "get": self._get,
            "update": self._update,
            "delete": self._delete,
            "create_collection": self._create_collection,
            "delete_collection": self._delete_collection,
            "list_collections": self._list_collections,
            "get_collection": self._get_collection,
            "count": self._count,
            "peek": self._peek,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                collection=collection,
                documents=documents or [],
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                query_text=query_text,
                query_embedding=query_embedding,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include or ["documents", "metadatas", "distances"],
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"Chroma error: {e}")
            return ToolResult.from_error(f"Chroma error: {e}")
    
    def _get_collection_obj(self, collection: str):
        """Get or create a collection."""
        client = self._get_client()
        embedding_fn = self._get_embedding_function()
        
        return client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_fn,
        )
    
    def _add(
        self,
        collection: Optional[str] = None,
        documents: Optional[list] = None,
        embeddings: Optional[list] = None,
        metadatas: Optional[list] = None,
        ids: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Add documents to a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not documents and not embeddings:
            return ToolResult.from_error("documents or embeddings required")
        
        coll = self._get_collection_obj(collection)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4())[:12] for _ in range(len(documents or embeddings))]
        
        add_kwargs = {"ids": ids}
        if documents:
            add_kwargs["documents"] = documents
        if embeddings:
            add_kwargs["embeddings"] = embeddings
        if metadatas:
            add_kwargs["metadatas"] = metadatas
        
        coll.add(**add_kwargs)
        
        return ToolResult.from_data({
            "collection": collection,
            "added": len(ids),
            "ids": ids,
        })
    
    def _query(
        self,
        collection: Optional[str] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[list] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Query the collection for similar documents."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not query_text and not query_embedding:
            return ToolResult.from_error("query_text or query_embedding required")
        
        coll = self._get_collection_obj(collection)
        
        query_kwargs = {
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        
        if query_text:
            query_kwargs["query_texts"] = [query_text]
        if query_embedding:
            query_kwargs["query_embeddings"] = [query_embedding]
        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document
        
        results = coll.query(**query_kwargs)
        
        # Format results
        formatted = []
        if results.get("ids"):
            for i, id_list in enumerate(results["ids"]):
                for j, doc_id in enumerate(id_list):
                    item = {"id": doc_id}
                    if "documents" in results and results["documents"]:
                        item["document"] = results["documents"][i][j]
                    if "metadatas" in results and results["metadatas"]:
                        item["metadata"] = results["metadatas"][i][j]
                    if "distances" in results and results["distances"]:
                        item["distance"] = results["distances"][i][j]
                    formatted.append(item)
        
        return ToolResult.from_data({
            "collection": collection,
            "results": formatted,
            "count": len(formatted),
        })
    
    def _get(
        self,
        collection: Optional[str] = None,
        ids: Optional[list] = None,
        where: Optional[dict] = None,
        include: Optional[list] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Get documents by ID or filter."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        coll = self._get_collection_obj(collection)
        
        get_kwargs = {
            "include": include or ["documents", "metadatas"],
        }
        if ids:
            get_kwargs["ids"] = ids
        if where:
            get_kwargs["where"] = where
        if limit:
            get_kwargs["limit"] = limit
        
        results = coll.get(**get_kwargs)
        
        # Format results
        formatted = []
        for i, doc_id in enumerate(results.get("ids", [])):
            item = {"id": doc_id}
            if results.get("documents"):
                item["document"] = results["documents"][i]
            if results.get("metadatas"):
                item["metadata"] = results["metadatas"][i]
            formatted.append(item)
        
        return ToolResult.from_data({
            "collection": collection,
            "documents": formatted,
            "count": len(formatted),
        })
    
    def _update(
        self,
        collection: Optional[str] = None,
        ids: Optional[list] = None,
        documents: Optional[list] = None,
        embeddings: Optional[list] = None,
        metadatas: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Update documents in a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not ids:
            return ToolResult.from_error("ids required")
        
        coll = self._get_collection_obj(collection)
        
        update_kwargs = {"ids": ids}
        if documents:
            update_kwargs["documents"] = documents
        if embeddings:
            update_kwargs["embeddings"] = embeddings
        if metadatas:
            update_kwargs["metadatas"] = metadatas
        
        coll.update(**update_kwargs)
        
        return ToolResult.from_data({
            "collection": collection,
            "updated": len(ids),
        })
    
    def _delete(
        self,
        collection: Optional[str] = None,
        ids: Optional[list] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete documents from a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        coll = self._get_collection_obj(collection)
        
        delete_kwargs = {}
        if ids:
            delete_kwargs["ids"] = ids
        if where:
            delete_kwargs["where"] = where
        if where_document:
            delete_kwargs["where_document"] = where_document
        
        coll.delete(**delete_kwargs)
        
        return ToolResult.from_data({
            "collection": collection,
            "deleted": True,
        })
    
    def _create_collection(
        self,
        collection: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        embedding_fn = self._get_embedding_function()
        
        coll = client.create_collection(
            name=collection,
            embedding_function=embedding_fn,
            metadata=metadata,
        )
        
        return ToolResult.from_data({
            "collection": collection,
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
        client.delete_collection(name=collection)
        
        return ToolResult.from_data({
            "collection": collection,
            "deleted": True,
        })
    
    def _list_collections(self, **kwargs) -> ToolResult:
        """List all collections."""
        client = self._get_client()
        collections = client.list_collections()
        
        return ToolResult.from_data({
            "collections": [
                {"name": c.name, "metadata": c.metadata}
                for c in collections
            ],
            "count": len(collections),
        })
    
    def _get_collection(
        self,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get collection info."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        coll = self._get_collection_obj(collection)
        
        return ToolResult.from_data({
            "name": coll.name,
            "count": coll.count(),
            "metadata": coll.metadata,
        })
    
    def _count(
        self,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get document count in a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        coll = self._get_collection_obj(collection)
        
        return ToolResult.from_data({
            "collection": collection,
            "count": coll.count(),
        })
    
    def _peek(
        self,
        collection: Optional[str] = None,
        limit: int = 10,
        **kwargs,
    ) -> ToolResult:
        """Peek at documents in a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        coll = self._get_collection_obj(collection)
        results = coll.peek(limit=limit)
        
        formatted = []
        for i, doc_id in enumerate(results.get("ids", [])):
            item = {"id": doc_id}
            if results.get("documents"):
                item["document"] = results["documents"][i]
            if results.get("metadatas"):
                item["metadata"] = results["metadatas"][i]
            formatted.append(item)
        
        return ToolResult.from_data({
            "collection": collection,
            "documents": formatted,
            "count": len(formatted),
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "query", "get", "update", "delete",
                                "create_collection", "delete_collection", "list_collections",
                                "get_collection", "count", "peek"],
                    },
                    "collection": {"type": "string"},
                    "documents": {"type": "array", "items": {"type": "string"}},
                    "metadatas": {"type": "array"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "query_text": {"type": "string"},
                    "n_results": {"type": "integer", "default": 10},
                    "where": {"type": "object"},
                },
                "required": ["action"],
            },
        }
