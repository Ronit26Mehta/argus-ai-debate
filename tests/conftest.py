"""
Test configuration and fixtures for ARGUS tests.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Generator


# ============== Mock LLM Response ==============

class MockLLMResponse:
    """Mock LLM response for testing."""
    
    def __init__(self, content: str = "Mock response"):
        self.content = content
        self.model = "mock-model"
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        self.finish_reason = "stop"


class MockLLM:
    """Mock LLM for testing without API calls."""
    
    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.calls = []
    
    def generate(self, prompt: str, **kwargs) -> MockLLMResponse:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return MockLLMResponse(response)
    
    def stream(self, prompt: str, **kwargs):
        yield MockLLMResponse("Streamed response")
    
    def embed(self, text: str) -> list[float]:
        return [0.1] * 384
    
    def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def mock_llm() -> MockLLM:
    """Provide a mock LLM for testing."""
    return MockLLM()


@pytest.fixture
def mock_llm_json() -> MockLLM:
    """Mock LLM that returns valid JSON."""
    return MockLLM([
        '{"claim": "Test claim", "confidence": 0.8, "polarity": "support"}',
        '{"verdict": "supported", "posterior": 0.75, "reasoning": "Test reasoning"}',
        '{"objectives": ["obj1", "obj2"], "domains": ["general"]}',
        '{"rebuttals": [{"target_id": "evid_123", "type": "logical", "content": "Test rebuttal", "strength": 0.6}]}',
    ])


# ============== Sample Documents ==============

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    from argus.core.models import Document, SourceType
    
    return Document(
        url="file:///test/document.txt",
        title="Test Document",
        content="This is a test document with some content. " * 50,
        source_type=SourceType.TEXT,
    )


@pytest.fixture
def sample_chunks(sample_document):
    """Create sample chunks from document."""
    from argus.core.models import Chunk
    
    return [
        Chunk(
            doc_id=sample_document.id,
            text=f"This is chunk {i} with test content.",
            start_char=i * 100,
            end_char=(i + 1) * 100,
            chunk_index=i,
        )
        for i in range(5)
    ]


# ============== C-DAG Fixtures ==============

@pytest.fixture
def empty_cdag():
    """Create an empty C-DAG."""
    from argus.cdag import CDAG
    return CDAG(name="test")


@pytest.fixture
def cdag_with_proposition(empty_cdag):
    """Create C-DAG with a proposition."""
    from argus.cdag import Proposition
    
    prop = Proposition(
        text="The treatment is effective",
        prior=0.5,
    )
    empty_cdag.add_proposition(prop)
    return empty_cdag, prop


@pytest.fixture
def cdag_with_evidence(cdag_with_proposition):
    """Create C-DAG with proposition and evidence."""
    from argus.cdag import Evidence, EvidenceType, EdgeType
    
    graph, prop = cdag_with_proposition
    
    support = Evidence(
        text="Clinical trial showed 25% improvement",
        evidence_type=EvidenceType.EMPIRICAL,
        polarity=1,
        confidence=0.85,
    )
    
    attack = Evidence(
        text="Side effects noted in 10% of patients",
        evidence_type=EvidenceType.EMPIRICAL,
        polarity=-1,
        confidence=0.7,
    )
    
    graph.add_evidence(support, prop.id, EdgeType.SUPPORTS)
    graph.add_evidence(attack, prop.id, EdgeType.ATTACKS)
    
    return graph, prop, support, attack


# ============== Configuration ==============

@pytest.fixture
def test_config():
    """Create test configuration."""
    from argus.core.config import ArgusConfig
    
    return ArgusConfig(
        default_provider="openai",
        default_model="gpt-4",
        temperature=0.5,
        max_tokens=1000,
    )


# ============== Temporary Files ==============

@pytest.fixture
def temp_text_file(tmp_path):
    """Create a temporary text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is test content for document ingestion.")
    return file_path


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file."""
    import json
    
    file_path = tmp_path / "test.json"
    data = {"key": "value", "items": [1, 2, 3]}
    file_path.write_text(json.dumps(data))
    return file_path


# ============== Provenance ==============

@pytest.fixture
def temp_ledger(tmp_path):
    """Create a temporary provenance ledger."""
    from argus.provenance import ProvenanceLedger
    
    ledger_path = tmp_path / "ledger.jsonl"
    return ProvenanceLedger(path=str(ledger_path))
