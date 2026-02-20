"""
Vertex AI Tools for ARGUS.

Search across private data stores and perform RAG operations.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class VertexAISearchTool(BaseTool):
    """
    Vertex AI Search - Search across private data stores.
    
    Features:
    - Search private data stores
    - Semantic and keyword search
    - Filtering and faceting
    - Search analytics
    
    Example:
        >>> tool = VertexAISearchTool(project_id="my-project", datastore_id="my-store")
        >>> result = tool(action="search", query="machine learning best practices")
    """
    
    name = "vertex_ai_search"
    description = "Search across your private, configured data stores in Vertex AI Search"
    category = ToolCategory.SEARCH
    version = "1.0.0"
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "global",
        datastore_id: Optional[str] = None,
        engine_id: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.datastore_id = datastore_id or os.getenv("VERTEX_DATASTORE_ID")
        self.engine_id = engine_id or os.getenv("VERTEX_ENGINE_ID")
        self._client = None
        
        logger.debug(f"VertexAISearch initialized for project {self.project_id}")
    
    def _get_client(self):
        """Lazy-load search client."""
        if self._client is None:
            try:
                from google.cloud import discoveryengine_v1 as discoveryengine
                self._client = discoveryengine.SearchServiceClient()
            except ImportError:
                raise ImportError("google-cloud-discoveryengine not installed. Run: pip install google-cloud-discoveryengine")
        return self._client
    
    def execute(
        self,
        action: str = "search",
        query: Optional[str] = None,
        page_size: int = 10,
        page_token: Optional[str] = None,
        filter_str: Optional[str] = None,
        order_by: Optional[str] = None,
        boost_spec: Optional[dict] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Vertex AI Search operations."""
        actions = {
            "search": self._search,
            "list_datastores": self._list_datastores,
            "get_datastore": self._get_datastore,
            "get_document": self._get_document,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                query=query,
                page_size=page_size,
                page_token=page_token,
                filter_str=filter_str,
                order_by=order_by,
                boost_spec=boost_spec,
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"VertexAISearch error: {e}")
            return ToolResult.from_error(f"VertexAISearch error: {e}")
    
    def _search(
        self,
        query: Optional[str] = None,
        page_size: int = 10,
        page_token: Optional[str] = None,
        filter_str: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Search the data store."""
        if not query:
            return ToolResult.from_error("query is required")
        
        if not self.datastore_id and not self.engine_id:
            return ToolResult.from_error("datastore_id or engine_id is required")
        
        from google.cloud import discoveryengine_v1 as discoveryengine
        
        client = self._get_client()
        
        # Build serving config path
        if self.engine_id:
            serving_config = f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/engines/{self.engine_id}/servingConfigs/default_serving_config"
        else:
            serving_config = f"projects/{self.project_id}/locations/{self.location}/dataStores/{self.datastore_id}/servingConfigs/default_serving_config"
        
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=page_size,
            page_token=page_token or "",
        )
        
        if filter_str:
            request.filter = filter_str
        
        response = client.search(request)
        
        results = []
        for result in response.results:
            doc = result.document
            results.append({
                "id": doc.id,
                "name": doc.name,
                "derived_struct_data": dict(doc.derived_struct_data) if doc.derived_struct_data else {},
                "struct_data": dict(doc.struct_data) if doc.struct_data else {},
            })
        
        return ToolResult.from_data({
            "query": query,
            "results": results,
            "total_size": response.total_size,
            "next_page_token": response.next_page_token,
        })
    
    def _list_datastores(self, **kwargs) -> ToolResult:
        """List available data stores."""
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine
            
            client = discoveryengine.DataStoreServiceClient()
            parent = f"projects/{self.project_id}/locations/{self.location}/collections/default_collection"
            
            datastores = []
            for ds in client.list_data_stores(parent=parent):
                datastores.append({
                    "name": ds.name,
                    "display_name": ds.display_name,
                    "industry_vertical": str(ds.industry_vertical),
                    "solution_types": [str(st) for st in ds.solution_types],
                })
            
            return ToolResult.from_data({
                "datastores": datastores,
                "count": len(datastores),
            })
        except Exception as e:
            return ToolResult.from_error(f"Failed to list datastores: {e}")
    
    def _get_datastore(
        self,
        datastore_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get data store details."""
        ds_id = datastore_id or self.datastore_id
        if not ds_id:
            return ToolResult.from_error("datastore_id is required")
        
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine
            
            client = discoveryengine.DataStoreServiceClient()
            name = f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{ds_id}"
            
            ds = client.get_data_store(name=name)
            
            return ToolResult.from_data({
                "name": ds.name,
                "display_name": ds.display_name,
                "industry_vertical": str(ds.industry_vertical),
            })
        except Exception as e:
            return ToolResult.from_error(f"Failed to get datastore: {e}")
    
    def _get_document(
        self,
        document_id: Optional[str] = None,
        datastore_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get a specific document."""
        if not document_id:
            return ToolResult.from_error("document_id is required")
        
        ds_id = datastore_id or self.datastore_id
        if not ds_id:
            return ToolResult.from_error("datastore_id is required")
        
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine
            
            client = discoveryengine.DocumentServiceClient()
            name = f"projects/{self.project_id}/locations/{self.location}/dataStores/{ds_id}/branches/default_branch/documents/{document_id}"
            
            doc = client.get_document(name=name)
            
            return ToolResult.from_data({
                "id": doc.id,
                "name": doc.name,
                "struct_data": dict(doc.struct_data) if doc.struct_data else {},
                "json_data": doc.json_data,
            })
        except Exception as e:
            return ToolResult.from_error(f"Failed to get document: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "list_datastores", "get_datastore", "get_document"],
                    },
                    "query": {"type": "string"},
                    "page_size": {"type": "integer", "default": 10},
                    "page_token": {"type": "string"},
                    "filter_str": {"type": "string"},
                    "document_id": {"type": "string"},
                },
                "required": ["action"],
            },
        }


class VertexAIRAGTool(BaseTool):
    """
    Vertex AI RAG Engine - Retrieval Augmented Generation.
    
    Features:
    - Private data retrieval
    - RAG corpus management
    - Document ingestion
    - Semantic retrieval
    
    Example:
        >>> tool = VertexAIRAGTool(project_id="my-project")
        >>> result = tool(action="retrieve", corpus_name="my-corpus", query="...")
    """
    
    name = "vertex_ai_rag"
    description = "Perform private data retrieval using Vertex AI RAG Engine"
    category = ToolCategory.SEARCH
    version = "1.0.0"
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        
        logger.debug(f"VertexAIRAG initialized for project {self.project_id}")
    
    def execute(
        self,
        action: str = "list_corpora",
        corpus_name: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 10,
        file_path: Optional[str] = None,
        document_name: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Vertex AI RAG operations."""
        actions = {
            "list_corpora": self._list_corpora,
            "create_corpus": self._create_corpus,
            "delete_corpus": self._delete_corpus,
            "retrieve": self._retrieve,
            "import_files": self._import_files,
            "list_files": self._list_files,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                corpus_name=corpus_name,
                query=query,
                top_k=top_k,
                file_path=file_path,
                document_name=document_name,
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"VertexAIRAG error: {e}")
            return ToolResult.from_error(f"VertexAIRAG error: {e}")
    
    def _list_corpora(self, **kwargs) -> ToolResult:
        """List RAG corpora."""
        try:
            from vertexai.preview import rag
            
            corpora = rag.list_corpora()
            
            corpus_list = []
            for corpus in corpora:
                corpus_list.append({
                    "name": corpus.name,
                    "display_name": corpus.display_name,
                    "description": corpus.description,
                })
            
            return ToolResult.from_data({
                "corpora": corpus_list,
                "count": len(corpus_list),
            })
        except ImportError:
            return ToolResult.from_error("vertexai not installed. Run: pip install google-cloud-aiplatform")
        except Exception as e:
            return ToolResult.from_error(f"Failed to list corpora: {e}")
    
    def _create_corpus(
        self,
        corpus_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new RAG corpus."""
        if not corpus_name:
            return ToolResult.from_error("corpus_name is required")
        
        try:
            from vertexai.preview import rag
            
            corpus = rag.create_corpus(
                display_name=corpus_name,
                description=description or "",
            )
            
            return ToolResult.from_data({
                "name": corpus.name,
                "display_name": corpus.display_name,
                "created": True,
            })
        except Exception as e:
            return ToolResult.from_error(f"Failed to create corpus: {e}")
    
    def _delete_corpus(
        self,
        corpus_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a RAG corpus."""
        if not corpus_name:
            return ToolResult.from_error("corpus_name is required")
        
        try:
            from vertexai.preview import rag
            
            rag.delete_corpus(name=corpus_name)
            
            return ToolResult.from_data({
                "corpus_name": corpus_name,
                "deleted": True,
            })
        except Exception as e:
            return ToolResult.from_error(f"Failed to delete corpus: {e}")
    
    def _retrieve(
        self,
        corpus_name: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 10,
        **kwargs,
    ) -> ToolResult:
        """Retrieve relevant documents from corpus."""
        if not corpus_name:
            return ToolResult.from_error("corpus_name is required")
        if not query:
            return ToolResult.from_error("query is required")
        
        try:
            from vertexai.preview import rag
            
            response = rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                text=query,
                similarity_top_k=top_k,
            )
            
            contexts = []
            for context in response.contexts.contexts:
                contexts.append({
                    "text": context.text,
                    "source": context.source_uri,
                    "score": context.score,
                })
            
            return ToolResult.from_data({
                "query": query,
                "contexts": contexts,
                "count": len(contexts),
            })
        except Exception as e:
            return ToolResult.from_error(f"Retrieval failed: {e}")
    
    def _import_files(
        self,
        corpus_name: Optional[str] = None,
        file_path: Optional[str] = None,
        gcs_uri: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Import files into a RAG corpus."""
        if not corpus_name:
            return ToolResult.from_error("corpus_name is required")
        if not file_path and not gcs_uri:
            return ToolResult.from_error("file_path or gcs_uri is required")
        
        try:
            from vertexai.preview import rag
            
            if gcs_uri:
                response = rag.import_files(
                    corpus_name=corpus_name,
                    paths=[gcs_uri],
                )
            else:
                # Upload local file to GCS first (simplified)
                return ToolResult.from_error("Local file upload not implemented. Use gcs_uri instead.")
            
            return ToolResult.from_data({
                "corpus_name": corpus_name,
                "imported": True,
                "imported_count": response.imported_rag_files_count,
            })
        except Exception as e:
            return ToolResult.from_error(f"Import failed: {e}")
    
    def _list_files(
        self,
        corpus_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List files in a corpus."""
        if not corpus_name:
            return ToolResult.from_error("corpus_name is required")
        
        try:
            from vertexai.preview import rag
            
            files = rag.list_files(corpus_name=corpus_name)
            
            file_list = []
            for f in files:
                file_list.append({
                    "name": f.name,
                    "display_name": f.display_name,
                    "size_bytes": f.size_bytes,
                })
            
            return ToolResult.from_data({
                "corpus_name": corpus_name,
                "files": file_list,
                "count": len(file_list),
            })
        except Exception as e:
            return ToolResult.from_error(f"Failed to list files: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_corpora", "create_corpus", "delete_corpus",
                                "retrieve", "import_files", "list_files"],
                    },
                    "corpus_name": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                    "file_path": {"type": "string"},
                    "gcs_uri": {"type": "string"},
                },
                "required": ["action"],
            },
        }
