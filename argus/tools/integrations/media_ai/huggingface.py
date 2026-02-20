"""
Hugging Face Tool for ARGUS.

ML model hub and inference API integration.
"""

from __future__ import annotations

import os
import logging
import base64
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class HuggingFaceTool(BaseTool):
    """
    Hugging Face - ML model hub and inference API.
    
    Features:
    - Model discovery and search
    - Inference API for various tasks
    - Dataset management
    - Space management
    - Repository management
    
    Example:
        >>> tool = HuggingFaceTool()
        >>> result = tool(action="inference", model="gpt2", inputs="Hello, I'm a language model")
        >>> result = tool(action="search_models", query="text-generation")
    """
    
    name = "huggingface"
    description = "ML model hub and inference API"
    category = ToolCategory.AI
    version = "1.0.0"
    
    API_BASE = "https://huggingface.co/api"
    INFERENCE_API = "https://api-inference.huggingface.co"
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_token = api_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        self._session = None
        
        if not self.api_token:
            logger.warning("No Hugging Face token provided")
        
        logger.debug("Hugging Face tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            self._session.headers.update(headers)
        return self._session
    
    def _request(
        self,
        method: str,
        url: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        raw_input: bool = False,
    ) -> dict | list | bytes:
        """Make API request."""
        session = self._get_session()
        
        json_data = None if raw_input else data
        raw_data = data if raw_input else None
        
        response = session.request(
            method=method,
            url=url,
            json=json_data,
            data=raw_data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("error", f"HTTP {response.status_code}")
                raise Exception(str(message))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        # Check if response is binary (image, audio, etc.)
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json() if response.text else {}
        elif "image" in content_type or "audio" in content_type:
            return response.content
        else:
            return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "search_models",
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        space: Optional[str] = None,
        query: Optional[str] = None,
        inputs: Optional[Any] = None,
        task: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Hugging Face operations."""
        actions = {
            # Model operations
            "search_models": self._search_models,
            "get_model": self._get_model,
            "list_model_files": self._list_model_files,
            
            # Dataset operations
            "search_datasets": self._search_datasets,
            "get_dataset": self._get_dataset,
            
            # Space operations
            "search_spaces": self._search_spaces,
            "get_space": self._get_space,
            
            # Inference
            "inference": self._inference,
            "text_generation": self._text_generation,
            "text_classification": self._text_classification,
            "token_classification": self._token_classification,
            "question_answering": self._question_answering,
            "summarization": self._summarization,
            "translation": self._translation,
            "fill_mask": self._fill_mask,
            "feature_extraction": self._feature_extraction,
            "image_classification": self._image_classification,
            "object_detection": self._object_detection,
            "image_to_text": self._image_to_text,
            "text_to_image": self._text_to_image,
            "text_to_speech": self._text_to_speech,
            "automatic_speech_recognition": self._asr,
            "zero_shot_classification": self._zero_shot_classification,
            "sentence_similarity": self._sentence_similarity,
            "conversational": self._conversational,
            
            # User operations
            "whoami": self._whoami,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                model=model,
                dataset=dataset,
                space=space,
                query=query,
                inputs=inputs,
                task=task,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
            return ToolResult.from_error(f"Hugging Face error: {e}")
    
    # Model operations
    def _search_models(
        self,
        query: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Search for models."""
        params = {"limit": min(limit, 100)}
        
        if query:
            params["search"] = query
        if task:
            params["pipeline_tag"] = task
        
        author = kwargs.get("author")
        if author:
            params["author"] = author
        
        sort = kwargs.get("sort", "downloads")
        params["sort"] = sort
        params["direction"] = kwargs.get("direction", "-1")
        
        response = self._request("GET", f"{self.API_BASE}/models", params=params)
        
        models = []
        for m in response if isinstance(response, list) else []:
            models.append({
                "id": m.get("id") or m.get("modelId"),
                "author": m.get("author"),
                "downloads": m.get("downloads"),
                "likes": m.get("likes"),
                "pipeline_tag": m.get("pipeline_tag"),
                "tags": m.get("tags", [])[:5],
                "lastModified": m.get("lastModified"),
            })
        
        return ToolResult.from_data({
            "models": models,
            "count": len(models),
        })
    
    def _get_model(
        self,
        model: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get model details."""
        if not model:
            return ToolResult.from_error("model is required")
        
        response = self._request("GET", f"{self.API_BASE}/models/{model}")
        
        return ToolResult.from_data({
            "model": {
                "id": response.get("id") or response.get("modelId"),
                "author": response.get("author"),
                "sha": response.get("sha"),
                "downloads": response.get("downloads"),
                "likes": response.get("likes"),
                "pipeline_tag": response.get("pipeline_tag"),
                "tags": response.get("tags", []),
                "cardData": response.get("cardData"),
                "siblings": [s.get("rfilename") for s in response.get("siblings", [])],
            }
        })
    
    def _list_model_files(
        self,
        model: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List model files."""
        if not model:
            return ToolResult.from_error("model is required")
        
        response = self._request("GET", f"{self.API_BASE}/models/{model}")
        
        files = []
        for s in response.get("siblings", []):
            files.append({
                "filename": s.get("rfilename"),
                "size": s.get("size"),
                "lfs": s.get("lfs"),
            })
        
        return ToolResult.from_data({
            "model": model,
            "files": files,
            "count": len(files),
        })
    
    # Dataset operations
    def _search_datasets(
        self,
        query: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Search for datasets."""
        params = {"limit": min(limit, 100)}
        
        if query:
            params["search"] = query
        
        author = kwargs.get("author")
        if author:
            params["author"] = author
        
        params["sort"] = kwargs.get("sort", "downloads")
        
        response = self._request("GET", f"{self.API_BASE}/datasets", params=params)
        
        datasets = []
        for d in response if isinstance(response, list) else []:
            datasets.append({
                "id": d.get("id"),
                "author": d.get("author"),
                "downloads": d.get("downloads"),
                "likes": d.get("likes"),
                "tags": d.get("tags", [])[:5],
            })
        
        return ToolResult.from_data({
            "datasets": datasets,
            "count": len(datasets),
        })
    
    def _get_dataset(
        self,
        dataset: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get dataset details."""
        if not dataset:
            return ToolResult.from_error("dataset is required")
        
        response = self._request("GET", f"{self.API_BASE}/datasets/{dataset}")
        
        return ToolResult.from_data({
            "dataset": {
                "id": response.get("id"),
                "author": response.get("author"),
                "downloads": response.get("downloads"),
                "likes": response.get("likes"),
                "tags": response.get("tags", []),
                "cardData": response.get("cardData"),
            }
        })
    
    # Space operations
    def _search_spaces(
        self,
        query: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Search for spaces."""
        params = {"limit": min(limit, 100)}
        
        if query:
            params["search"] = query
        
        author = kwargs.get("author")
        if author:
            params["author"] = author
        
        params["sort"] = kwargs.get("sort", "likes")
        
        response = self._request("GET", f"{self.API_BASE}/spaces", params=params)
        
        spaces = []
        for s in response if isinstance(response, list) else []:
            spaces.append({
                "id": s.get("id"),
                "author": s.get("author"),
                "likes": s.get("likes"),
                "sdk": s.get("sdk"),
                "tags": s.get("tags", [])[:5],
            })
        
        return ToolResult.from_data({
            "spaces": spaces,
            "count": len(spaces),
        })
    
    def _get_space(
        self,
        space: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get space details."""
        if not space:
            return ToolResult.from_error("space is required")
        
        response = self._request("GET", f"{self.API_BASE}/spaces/{space}")
        
        return ToolResult.from_data({
            "space": {
                "id": response.get("id"),
                "author": response.get("author"),
                "likes": response.get("likes"),
                "sdk": response.get("sdk"),
                "tags": response.get("tags", []),
                "runtime": response.get("runtime"),
            }
        })
    
    # Inference operations
    def _inference(
        self,
        model: Optional[str] = None,
        inputs: Optional[Any] = None,
        **kwargs,
    ) -> ToolResult:
        """Run inference on a model."""
        if not model:
            return ToolResult.from_error("model is required")
        if inputs is None:
            return ToolResult.from_error("inputs is required")
        
        url = f"{self.INFERENCE_API}/models/{model}"
        
        data = {"inputs": inputs}
        
        parameters = kwargs.get("parameters")
        if parameters:
            data["parameters"] = parameters
        
        options = kwargs.get("options")
        if options:
            data["options"] = options
        
        response = self._request("POST", url, data=data)
        
        # Handle binary responses (images, audio)
        if isinstance(response, bytes):
            return ToolResult.from_data({
                "output_base64": base64.b64encode(response).decode("utf-8"),
                "output_type": "binary",
            })
        
        return ToolResult.from_data({
            "output": response,
        })
    
    def _text_generation(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate text."""
        model = model or "gpt2"
        
        if not inputs:
            return ToolResult.from_error("inputs (prompt) is required")
        
        parameters = {
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.95),
            "do_sample": kwargs.get("do_sample", True),
        }
        
        return self._inference(
            model=model,
            inputs=inputs,
            parameters=parameters,
            **kwargs,
        )
    
    def _text_classification(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Classify text."""
        model = model or "distilbert-base-uncased-finetuned-sst-2-english"
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _token_classification(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Token classification (NER, POS tagging)."""
        model = model or "dbmdz/bert-large-cased-finetuned-conll03-english"
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _question_answering(
        self,
        model: Optional[str] = None,
        inputs: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Answer questions."""
        model = model or "deepset/roberta-base-squad2"
        
        if not inputs:
            question = kwargs.get("question")
            context = kwargs.get("context")
            if not question or not context:
                return ToolResult.from_error("inputs with 'question' and 'context' required")
            inputs = {"question": question, "context": context}
        
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _summarization(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Summarize text."""
        model = model or "facebook/bart-large-cnn"
        
        parameters = {
            "max_length": kwargs.get("max_length", 150),
            "min_length": kwargs.get("min_length", 30),
        }
        
        return self._inference(model=model, inputs=inputs, parameters=parameters, **kwargs)
    
    def _translation(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Translate text."""
        model = model or "Helsinki-NLP/opus-mt-en-de"
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _fill_mask(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Fill masked tokens."""
        model = model or "bert-base-uncased"
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _feature_extraction(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Extract features/embeddings."""
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _image_classification(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Classify images."""
        model = model or "google/vit-base-patch16-224"
        
        # Inputs should be base64 encoded image
        if inputs and not inputs.startswith("http"):
            # Assume base64, decode for binary upload
            try:
                image_bytes = base64.b64decode(inputs)
                url = f"{self.INFERENCE_API}/models/{model}"
                
                session = self._get_session()
                response = session.post(
                    url,
                    data=image_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                )
                
                if response.status_code >= 400:
                    raise Exception(f"HTTP {response.status_code}")
                
                return ToolResult.from_data({"output": response.json()})
            except Exception:
                pass
        
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _object_detection(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Detect objects in images."""
        model = model or "facebook/detr-resnet-50"
        return self._image_classification(model=model, inputs=inputs, **kwargs)
    
    def _image_to_text(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate text from images (captioning, OCR)."""
        model = model or "Salesforce/blip-image-captioning-base"
        return self._image_classification(model=model, inputs=inputs, **kwargs)
    
    def _text_to_image(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate images from text."""
        model = model or "stabilityai/stable-diffusion-2-1"
        
        if not inputs:
            return ToolResult.from_error("inputs (prompt) is required")
        
        url = f"{self.INFERENCE_API}/models/{model}"
        
        response = self._request("POST", url, data={"inputs": inputs})
        
        if isinstance(response, bytes):
            return ToolResult.from_data({
                "image_base64": base64.b64encode(response).decode("utf-8"),
                "format": "png",
            })
        
        return ToolResult.from_data({"output": response})
    
    def _text_to_speech(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate speech from text."""
        model = model or "espnet/kan-bayashi_ljspeech_vits"
        
        if not inputs:
            return ToolResult.from_error("inputs (text) is required")
        
        url = f"{self.INFERENCE_API}/models/{model}"
        
        response = self._request("POST", url, data={"inputs": inputs})
        
        if isinstance(response, bytes):
            return ToolResult.from_data({
                "audio_base64": base64.b64encode(response).decode("utf-8"),
                "format": "wav",
            })
        
        return ToolResult.from_data({"output": response})
    
    def _asr(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Automatic speech recognition."""
        model = model or "openai/whisper-small"
        
        if not inputs:
            return ToolResult.from_error("inputs (audio base64) is required")
        
        try:
            audio_bytes = base64.b64decode(inputs)
            url = f"{self.INFERENCE_API}/models/{model}"
            
            session = self._get_session()
            response = session.post(
                url,
                data=audio_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            
            if response.status_code >= 400:
                raise Exception(f"HTTP {response.status_code}")
            
            return ToolResult.from_data({"output": response.json()})
        except Exception as e:
            return ToolResult.from_error(f"ASR error: {e}")
    
    def _zero_shot_classification(
        self,
        model: Optional[str] = None,
        inputs: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Zero-shot classification."""
        model = model or "facebook/bart-large-mnli"
        
        candidate_labels = kwargs.get("candidate_labels", [])
        if not candidate_labels:
            return ToolResult.from_error("candidate_labels is required")
        
        data = {
            "inputs": inputs,
            "parameters": {"candidate_labels": candidate_labels},
        }
        
        url = f"{self.INFERENCE_API}/models/{model}"
        response = self._request("POST", url, data=data)
        
        return ToolResult.from_data({"output": response})
    
    def _sentence_similarity(
        self,
        model: Optional[str] = None,
        inputs: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Compute sentence similarity."""
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        
        if not inputs:
            source = kwargs.get("source_sentence")
            sentences = kwargs.get("sentences", [])
            if not source or not sentences:
                return ToolResult.from_error("source_sentence and sentences required")
            inputs = {"source_sentence": source, "sentences": sentences}
        
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    def _conversational(
        self,
        model: Optional[str] = None,
        inputs: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Conversational response."""
        model = model or "microsoft/DialoGPT-medium"
        
        if not inputs:
            text = kwargs.get("text")
            past_user_inputs = kwargs.get("past_user_inputs", [])
            generated_responses = kwargs.get("generated_responses", [])
            
            if not text:
                return ToolResult.from_error("text is required")
            
            inputs = {
                "text": text,
                "past_user_inputs": past_user_inputs,
                "generated_responses": generated_responses,
            }
        
        return self._inference(model=model, inputs=inputs, **kwargs)
    
    # User operations
    def _whoami(self, **kwargs) -> ToolResult:
        """Get current user info."""
        if not self.api_token:
            return ToolResult.from_error("Authentication required")
        
        response = self._request("GET", f"{self.API_BASE}/whoami")
        
        return ToolResult.from_data({
            "user": {
                "name": response.get("name"),
                "fullname": response.get("fullname"),
                "email": response.get("email"),
                "type": response.get("type"),
                "orgs": response.get("orgs", []),
            }
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
                            "search_models", "get_model", "list_model_files",
                            "search_datasets", "get_dataset",
                            "search_spaces", "get_space",
                            "inference", "text_generation", "text_classification",
                            "token_classification", "question_answering",
                            "summarization", "translation", "fill_mask",
                            "feature_extraction", "image_classification",
                            "object_detection", "image_to_text", "text_to_image",
                            "text_to_speech", "automatic_speech_recognition",
                            "zero_shot_classification", "sentence_similarity",
                            "conversational", "whoami",
                        ],
                    },
                    "model": {"type": "string"},
                    "dataset": {"type": "string"},
                    "space": {"type": "string"},
                    "query": {"type": "string"},
                    "inputs": {"type": ["string", "object", "array"]},
                    "task": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
