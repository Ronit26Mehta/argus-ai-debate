# ARGUS Core Module

## Overview

The `core/` module contains configuration, LLM integrations, and shared utilities for the ARGUS framework.

## Structure

```
core/
├── __init__.py
├── config.py          # Configuration classes
├── models.py          # Data models
└── llm/               # 27+ LLM providers
    ├── __init__.py
    ├── base.py        # BaseLLM interface
    ├── registry.py    # Provider registry
    ├── openai.py      # OpenAI
    ├── anthropic.py   # Anthropic Claude
    ├── gemini.py      # Google Gemini
    ├── ollama.py      # Local Ollama
    └── ...            # 20+ more providers
```

---

## LLM Providers (27+)

### Available Providers

| Provider | Model Examples | API Key Env |
|----------|---------------|-------------|
| `openai` | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| `anthropic` | claude-3-5-sonnet | `ANTHROPIC_API_KEY` |
| `gemini` | gemini-1.5-pro | `GOOGLE_API_KEY` |
| `ollama` | llama3.2, mistral | None (local) |
| `cohere` | command-r-plus | `COHERE_API_KEY` |
| `mistral` | mistral-large | `MISTRAL_API_KEY` |
| `groq` | llama-3.1-70b | `GROQ_API_KEY` |
| `deepseek` | deepseek-chat | `DEEPSEEK_API_KEY` |
| `xai` | grok-beta | `XAI_API_KEY` |
| `perplexity` | sonar-pro | `PERPLEXITY_API_KEY` |
| `together` | meta-llama/Llama-3 | `TOGETHER_API_KEY` |
| `fireworks` | llama-v3p1-70b | `FIREWORKS_API_KEY` |
| `nvidia` | meta/llama3-70b | `NVIDIA_API_KEY` |
| `azure` | gpt-4 (Azure) | `AZURE_OPENAI_API_KEY` |
| `bedrock` | anthropic.claude-3 | AWS credentials |
| `vertex` | gemini-1.5-pro | GCP credentials |
| `huggingface` | meta-llama/Llama | `HF_TOKEN` |
| `watsonx` | ibm/granite | `WATSONX_API_KEY` |
| `databricks` | dbrx-instruct | `DATABRICKS_TOKEN` |
| `sambanova` | Meta-Llama-3.1 | `SAMBANOVA_API_KEY` |
| `cerebras` | llama3.1-70b | `CEREBRAS_API_KEY` |
| `cloudflare` | @cf/meta/llama | `CLOUDFLARE_API_TOKEN` |
| `replicate` | meta/llama-3 | `REPLICATE_API_TOKEN` |
| `vllm` | (self-hosted) | None (local) |
| `llamacpp` | (local GGUF) | None (local) |
| `litellm` | (proxy) | Various |

### Usage Examples

```python
from argus.core.llm import get_llm, list_providers, LLMRegistry

# List all providers
providers = list_providers()
print(providers)

# Basic generation
llm = get_llm("openai", model="gpt-4o")
response = llm.generate("What is the meaning of life?")
print(response.content)
print(f"Tokens used: {response.usage.total_tokens}")

# With custom parameters
llm = get_llm(
    "anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=2000,
)
response = llm.generate(
    prompt="Write a poem about AI",
    system="You are a creative poet.",
)

# Streaming
llm = get_llm("openai", model="gpt-4o-mini")
for chunk in llm.stream("Tell me a short story"):
    print(chunk, end="", flush=True)

# Chat with messages
from argus.core.llm import Message

messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is Python?"),
    Message(role="assistant", content="Python is a programming language."),
    Message(role="user", content="What can I build with it?"),
]
response = llm.generate(messages)

# Embeddings (if supported)
llm = get_llm("openai", model="gpt-4o")
vectors = llm.embed(["Hello world", "AI is cool"])
print(f"Embedding dimension: {len(vectors[0])}")
```

### Local Providers (No API Key)

```python
# Ollama (requires Ollama installed)
llm = get_llm("ollama", model="llama3.2")
response = llm.generate("Hello!")

# Llama.cpp (requires model file)
llm = get_llm("llamacpp", model_path="/path/to/model.gguf")
response = llm.generate("What is 2+2?")

# vLLM (requires vLLM server running)
llm = get_llm("vllm", base_url="http://localhost:8000/v1")
response = llm.generate("Hello!")
```

---

## Configuration

### ArgusConfig

```python
from argus.core.config import ArgusConfig, LLMProviderConfig

# Load from environment
config = ArgusConfig()

# Access settings
print(config.llm.openai_api_key)
print(config.llm.default_provider)
print(config.llm.get_available_providers())

# Custom configuration
config = ArgusConfig(
    default_provider="anthropic",
    default_model="claude-3-5-sonnet-20241022",
    temperature=0.8,
)
```

### Environment Variables

```env
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Defaults
ARGUS_DEFAULT_PROVIDER=openai
ARGUS_DEFAULT_MODEL=gpt-4o
ARGUS_TEMPERATURE=0.7
```

---

## Data Models

```python
from argus.core.models import (
    Chunk,
    Document,
    Embedding,
    Evidence,
    Argument,
)

# Create a document
doc = Document(
    id="doc-001",
    title="My Paper",
    content="Full text content...",
    metadata={"author": "John Doe"},
)

# Create a chunk
chunk = Chunk(
    id="chunk-001",
    text="This is a text chunk.",
    document_id="doc-001",
    start=0,
    end=100,
)

# Create embedding
embedding = Embedding(
    source_id="chunk-001",
    vector=[0.1, 0.2, 0.3, ...],
    model="all-MiniLM-L6-v2",
)
```

---

## Extending with Custom Providers

```python
from argus.core.llm import BaseLLM, LLMRegistry, LLMResponse

class MyCustomLLM(BaseLLM):
    """Custom LLM provider."""
    
    def __init__(self, model: str = "my-model", api_key: str = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
    
    def generate(self, prompt, **kwargs) -> LLMResponse:
        # Your implementation
        return LLMResponse(content="Response", model=self.model)
    
    def stream(self, prompt, **kwargs):
        yield "Streaming "
        yield "response"

# Register
LLMRegistry.register("my_provider", MyCustomLLM)

# Use
llm = get_llm("my_provider", model="my-model")
```
