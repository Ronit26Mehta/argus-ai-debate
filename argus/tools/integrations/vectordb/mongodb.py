"""
MongoDB Tool for ARGUS.

Document database with vector search capabilities for AI agents.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class MongoDBTool(BaseTool):
    """
    MongoDB - Flexible document database with vector search.
    
    Features:
    - Document CRUD operations
    - Atlas Vector Search integration
    - Aggregation pipelines
    - Index management
    - Collection management
    
    Example:
        >>> tool = MongoDBTool(connection_string="mongodb://localhost:27017")
        >>> result = tool(action="insert", database="mydb", collection="docs", documents=[...])
        >>> result = tool(action="find", database="mydb", collection="docs", query={...})
    """
    
    name = "mongodb"
    description = "Document database with vector search capabilities"
    category = ToolCategory.DATA
    version = "1.0.0"
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        database: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.connection_string = connection_string or os.getenv(
            "MONGODB_CONNECTION_STRING",
            "mongodb://localhost:27017"
        )
        self.default_database = database or os.getenv("MONGODB_DATABASE", "argus")
        
        self._client = None
        
        logger.debug(f"MongoDB initialized (db={self.default_database})")
    
    def _get_client(self):
        """Lazy-load MongoDB client."""
        if self._client is None:
            try:
                from pymongo import MongoClient
                self._client = MongoClient(self.connection_string)
            except ImportError:
                raise ImportError("pymongo not installed. Run: pip install pymongo")
        return self._client
    
    def execute(
        self,
        action: str = "list_databases",
        database: Optional[str] = None,
        collection: Optional[str] = None,
        documents: Optional[list] = None,
        document: Optional[dict] = None,
        query: Optional[dict] = None,
        update: Optional[dict] = None,
        projection: Optional[dict] = None,
        sort: Optional[list] = None,
        limit: int = 100,
        skip: int = 0,
        vector: Optional[list] = None,
        index_name: str = "vector_index",
        num_candidates: int = 100,
        pipeline: Optional[list] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute MongoDB operations."""
        actions = {
            "insert": self._insert,
            "insert_one": self._insert_one,
            "find": self._find,
            "find_one": self._find_one,
            "update": self._update,
            "update_one": self._update_one,
            "delete": self._delete,
            "delete_one": self._delete_one,
            "count": self._count,
            "aggregate": self._aggregate,
            "vector_search": self._vector_search,
            "create_index": self._create_index,
            "create_vector_index": self._create_vector_index,
            "list_indexes": self._list_indexes,
            "drop_index": self._drop_index,
            "list_collections": self._list_collections,
            "create_collection": self._create_collection,
            "drop_collection": self._drop_collection,
            "list_databases": self._list_databases,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            db_name = database or self.default_database
            return actions[action](
                database=db_name,
                collection=collection,
                documents=documents or [],
                document=document or {},
                query=query or {},
                update=update or {},
                projection=projection,
                sort=sort,
                limit=limit,
                skip=skip,
                vector=vector,
                index_name=index_name,
                num_candidates=num_candidates,
                pipeline=pipeline,
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"MongoDB error: {e}")
            return ToolResult.from_error(f"MongoDB error: {e}")
    
    def _insert(
        self,
        database: str,
        collection: Optional[str] = None,
        documents: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Insert multiple documents."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not documents:
            return ToolResult.from_error("documents list is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        # Add timestamps if not present
        for doc in documents:
            if "_created_at" not in doc:
                doc["_created_at"] = datetime.utcnow()
        
        result = coll.insert_many(documents)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "inserted_count": len(result.inserted_ids),
            "inserted_ids": [str(id_) for id_ in result.inserted_ids],
        })
    
    def _insert_one(
        self,
        database: str,
        collection: Optional[str] = None,
        document: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Insert a single document."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not document:
            return ToolResult.from_error("document is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        if "_created_at" not in document:
            document["_created_at"] = datetime.utcnow()
        
        result = coll.insert_one(document)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "inserted_id": str(result.inserted_id),
        })
    
    def _find(
        self,
        database: str,
        collection: Optional[str] = None,
        query: Optional[dict] = None,
        projection: Optional[dict] = None,
        sort: Optional[list] = None,
        limit: int = 100,
        skip: int = 0,
        **kwargs,
    ) -> ToolResult:
        """Find documents matching query."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        cursor = coll.find(query or {}, projection=projection)
        
        if sort:
            # Sort format: [["field", 1], ["field2", -1]]
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        
        documents = []
        for doc in cursor:
            # Convert ObjectId to string
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            documents.append(doc)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "documents": documents,
            "count": len(documents),
        })
    
    def _find_one(
        self,
        database: str,
        collection: Optional[str] = None,
        query: Optional[dict] = None,
        projection: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Find a single document."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        doc = coll.find_one(query or {}, projection=projection)
        
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "document": doc,
            "found": doc is not None,
        })
    
    def _update(
        self,
        database: str,
        collection: Optional[str] = None,
        query: Optional[dict] = None,
        update: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Update multiple documents."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not update:
            return ToolResult.from_error("update is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        # Ensure we have proper update operators
        if not any(k.startswith("$") for k in update.keys()):
            update = {"$set": update}
        
        # Add updated_at timestamp
        if "$set" in update:
            update["$set"]["_updated_at"] = datetime.utcnow()
        else:
            update["$set"] = {"_updated_at": datetime.utcnow()}
        
        result = coll.update_many(query or {}, update)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
        })
    
    def _update_one(
        self,
        database: str,
        collection: Optional[str] = None,
        query: Optional[dict] = None,
        update: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a single document."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not update:
            return ToolResult.from_error("update is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        if not any(k.startswith("$") for k in update.keys()):
            update = {"$set": update}
        
        if "$set" in update:
            update["$set"]["_updated_at"] = datetime.utcnow()
        else:
            update["$set"] = {"_updated_at": datetime.utcnow()}
        
        result = coll.update_one(query or {}, update)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
        })
    
    def _delete(
        self,
        database: str,
        collection: Optional[str] = None,
        query: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete multiple documents."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        result = coll.delete_many(query or {})
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "deleted_count": result.deleted_count,
        })
    
    def _delete_one(
        self,
        database: str,
        collection: Optional[str] = None,
        query: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a single document."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        result = coll.delete_one(query or {})
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "deleted_count": result.deleted_count,
        })
    
    def _count(
        self,
        database: str,
        collection: Optional[str] = None,
        query: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Count documents matching query."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        count = coll.count_documents(query or {})
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "count": count,
        })
    
    def _aggregate(
        self,
        database: str,
        collection: Optional[str] = None,
        pipeline: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Run aggregation pipeline."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not pipeline:
            return ToolResult.from_error("pipeline is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        results = list(coll.aggregate(pipeline))
        
        # Convert ObjectIds
        for doc in results:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "results": results,
            "count": len(results),
        })
    
    def _vector_search(
        self,
        database: str,
        collection: Optional[str] = None,
        vector: Optional[list] = None,
        index_name: str = "vector_index",
        limit: int = 10,
        num_candidates: int = 100,
        projection: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Run Atlas Vector Search."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        if not vector:
            return ToolResult.from_error("vector is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": num_candidates,
                    "limit": limit,
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        if projection:
            pipeline.append({"$project": projection})
        
        results = list(coll.aggregate(pipeline))
        
        # Convert ObjectIds
        for doc in results:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "results": results,
            "count": len(results),
        })
    
    def _create_index(
        self,
        database: str,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create an index on a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        keys = kwargs.get("keys", [("_id", 1)])
        unique = kwargs.get("unique", False)
        name = kwargs.get("name")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        index_name = coll.create_index(keys, unique=unique, name=name)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "index_name": index_name,
        })
    
    def _create_vector_index(
        self,
        database: str,
        collection: Optional[str] = None,
        index_name: str = "vector_index",
        **kwargs,
    ) -> ToolResult:
        """Create a vector search index (Atlas)."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        dimension = kwargs.get("dimension", 1536)
        path = kwargs.get("path", "embedding")
        similarity = kwargs.get("similarity", "cosine")
        
        # Note: Vector indexes need to be created via Atlas UI or Atlas API
        # This is a helper that documents the required configuration
        
        index_definition = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": dimension,
                        "path": path,
                        "similarity": similarity,
                    }
                ]
            }
        }
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "index_definition": index_definition,
            "note": "Create this index via MongoDB Atlas Search or Atlas CLI",
        })
    
    def _list_indexes(
        self,
        database: str,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List indexes on a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        indexes = list(coll.list_indexes())
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "indexes": indexes,
            "count": len(indexes),
        })
    
    def _drop_index(
        self,
        database: str,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Drop an index."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        index_name = kwargs.get("name")
        if not index_name:
            return ToolResult.from_error("index name is required")
        
        client = self._get_client()
        db = client[database]
        coll = db[collection]
        
        coll.drop_index(index_name)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "dropped": index_name,
        })
    
    def _list_collections(
        self,
        database: str,
        **kwargs,
    ) -> ToolResult:
        """List collections in database."""
        client = self._get_client()
        db = client[database]
        
        collections = db.list_collection_names()
        
        return ToolResult.from_data({
            "database": database,
            "collections": collections,
            "count": len(collections),
        })
    
    def _create_collection(
        self,
        database: str,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        
        db.create_collection(collection)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "created": True,
        })
    
    def _drop_collection(
        self,
        database: str,
        collection: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Drop a collection."""
        if not collection:
            return ToolResult.from_error("collection name is required")
        
        client = self._get_client()
        db = client[database]
        
        db.drop_collection(collection)
        
        return ToolResult.from_data({
            "database": database,
            "collection": collection,
            "dropped": True,
        })
    
    def _list_databases(self, **kwargs) -> ToolResult:
        """List all databases."""
        client = self._get_client()
        
        databases = client.list_database_names()
        
        return ToolResult.from_data({
            "databases": databases,
            "count": len(databases),
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "insert", "insert_one", "find", "find_one",
                            "update", "update_one", "delete", "delete_one",
                            "count", "aggregate", "vector_search",
                            "create_index", "create_vector_index", "list_indexes", "drop_index",
                            "list_collections", "create_collection", "drop_collection",
                            "list_databases"
                        ],
                    },
                    "database": {"type": "string"},
                    "collection": {"type": "string"},
                    "documents": {"type": "array"},
                    "document": {"type": "object"},
                    "query": {"type": "object"},
                    "update": {"type": "object"},
                    "projection": {"type": "object"},
                    "sort": {"type": "array"},
                    "limit": {"type": "integer", "default": 100},
                    "skip": {"type": "integer", "default": 0},
                    "vector": {"type": "array", "items": {"type": "number"}},
                    "index_name": {"type": "string"},
                    "pipeline": {"type": "array"},
                },
                "required": ["action"],
            },
        }
