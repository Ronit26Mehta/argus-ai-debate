"""
ARGUS - Agentic Research & Governance Unified System.

A debate-native, multi-agent AI framework for evidence-based reasoning
with structured argumentation, decision-theoretic planning, and provenance.

Quick Start:
    >>> from argus import CDAG, Proposition, Evidence, RDCOrchestrator
    >>> 
    >>> # Create a debate graph
    >>> graph = CDAG()
    >>> prop = Proposition(text="The treatment is effective", prior=0.5)
    >>> graph.add_proposition(prop)
    >>> 
    >>> # Run a full debate
    >>> orchestrator = RDCOrchestrator()
    >>> result = orchestrator.debate("Drug X reduces symptoms by >20%")
    >>> print(f"Verdict: {result.verdict.label}")

Architecture:
    1. Knowledge & Data Layer - Document ingestion, chunking, embeddings
    2. Retrieval & Evidence - Hybrid search, reranking, cite & critique
    3. Argumentation (C-DAG) - Conceptual Debate Graph with propagation
    4. Decision & Experimentation - Bayesian updates, EIG planning
    5. Provenance & Governance - PROV-O ledger, integrity checks
    6. Agent Orchestration - Moderator, Specialists, Refuter, Jury
"""

__version__ = "1.4.2"
__author__ = "ARGUS Team"
__license__ = "MIT"

# Core configuration
from argus.core.config import ArgusConfig, get_config

# Tools framework (v1.1.0)
from argus.tools import (
    BaseTool,
    ToolResult,
    ToolRegistry,
    ToolExecutor,
    ResultCache,
    Guardrail,
    register_tool,
    get_tool,
    list_tools,
)

# Outputs/Reports (v1.1.0)
from argus.outputs import (
    ReportGenerator,
    DebateReport,
    ReportConfig,
    export_json,
    export_markdown,
)

# Metrics & Traces (v1.1.0)
from argus.metrics import (
    MetricsCollector,
    Tracer,
    record_counter,
    record_gauge,
    record_histogram,
    get_tracer,
)

# Connectors (v1.1.0)
from argus.knowledge.connectors import (
    BaseConnector,
    ConnectorRegistry,
    WebConnector,
    register_connector,
)

# Core data models
from argus.core.models import (
    Document,
    Chunk,
    Embedding,
    Claim,
    Citation,
    SourceType,
)

# LLM providers
from argus.core.llm import (
    BaseLLM,
    Message,
    LLMResponse,
    get_llm,
    list_providers,
)

# C-DAG (Conceptual Debate Graph)
from argus.cdag import (
    CDAG,
    Proposition,
    Evidence,
    Rebuttal,
    Finding,
    Assumption,
    Edge,
    EdgeType,
    EdgePolarity,
    NodeStatus,
    propagate_influence,
    compute_posterior,
)

# Decision layer
from argus.decision import (
    BayesianUpdater,
    EIGEstimator,
    VoIPlanner,
    CalibrationMetrics,
    compute_brier_score,
    compute_ece,
    temperature_scaling,
)

# Knowledge layer
from argus.knowledge import (
    DocumentLoader,
    Chunker,
    ChunkingStrategy,
    EmbeddingGenerator,
    HybridIndex,
)

# Retrieval layer
from argus.retrieval import (
    HybridRetriever,
    CrossEncoderReranker,
    cite_and_critique,
)

# Agents
from argus.agents import (
    Moderator,
    Specialist,
    Refuter,
    Jury,
    Verdict,
)

# Provenance
from argus.provenance import (
    ProvenanceLedger,
    ProvenanceEvent,
    EventType,
)

# Orchestrator
from argus.orchestrator import RDCOrchestrator, DebateResult

# Human-in-the-Loop (v1.4.0)
from argus.hitl import (
    HITLConfig,
    HITLMiddleware,
    InterruptRequest,
    InterruptStatus,
    ApprovalHandler,
    RejectionHandler,
    DecisionRouter,
    FeedbackCallback,
)

# Memory Systems (v1.4.0)
from argus.memory import (
    MemoryConfig,
    ConversationBufferMemory,
    ConversationWindowMemory,
    VectorStoreMemory,
    SemanticCache,
    MemoryStore,
    SQLiteStore,
)

# MCP Integration (v1.4.0)
from argus.mcp import (
    MCPServerConfig,
    MCPClientConfig,
    ArgusServer,
    MCPClient,
    ToolAdapter,
    ResourceRegistry,
)

# Durable Execution (v1.4.0)
from argus.durable import (
    DurableConfig,
    DurableWorkflow,
    Checkpoint,
    MemoryCheckpointer,
    SQLiteCheckpointer,
    StateManager,
    idempotent_task,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    # Config
    "ArgusConfig",
    "get_config",
    # Tools (v1.1.0)
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
    "ResultCache",
    "Guardrail",
    "register_tool",
    "get_tool",
    "list_tools",
    # Outputs (v1.1.0)
    "ReportGenerator",
    "DebateReport",
    "ReportConfig",
    "export_json",
    "export_markdown",
    # Metrics (v1.1.0)
    "MetricsCollector",
    "Tracer",
    "record_counter",
    "record_gauge",
    "record_histogram",
    "get_tracer",
    # Connectors (v1.1.0)
    "BaseConnector",
    "ConnectorRegistry",
    "WebConnector",
    "register_connector",
    # Models
    "Document",
    "Chunk",
    "Embedding",
    "Claim",
    "Citation",
    "SourceType",
    # LLM
    "BaseLLM",
    "Message",
    "LLMResponse",
    "get_llm",
    "list_providers",
    # C-DAG
    "CDAG",
    "Proposition",
    "Evidence",
    "Rebuttal",
    "Finding",
    "Assumption",
    "Edge",
    "EdgeType",
    "EdgePolarity",
    "NodeStatus",
    "propagate_influence",
    "compute_posterior",
    # Decision
    "BayesianUpdater",
    "EIGEstimator",
    "VoIPlanner",
    "CalibrationMetrics",
    "compute_brier_score",
    "compute_ece",
    "temperature_scaling",
    # Knowledge
    "DocumentLoader",
    "Chunker",
    "ChunkingStrategy",
    "EmbeddingGenerator",
    "HybridIndex",
    # Retrieval
    "HybridRetriever",
    "CrossEncoderReranker",
    "cite_and_critique",
    # Agents
    "Moderator",
    "Specialist",
    "Refuter",
    "Jury",
    "Verdict",
    # Provenance
    "ProvenanceLedger",
    "ProvenanceEvent",
    "EventType",
    # Orchestrator
    "RDCOrchestrator",
    "DebateResult",
    # HITL (v1.4.0)
    "HITLConfig",
    "HITLMiddleware",
    "InterruptRequest",
    "InterruptStatus",
    "ApprovalHandler",
    "RejectionHandler",
    "DecisionRouter",
    "FeedbackCallback",
    # Memory (v1.4.0)
    "MemoryConfig",
    "ConversationBufferMemory",
    "ConversationWindowMemory",
    "VectorStoreMemory",
    "SemanticCache",
    "MemoryStore",
    "SQLiteStore",
    # MCP (v1.4.0)
    "MCPServerConfig",
    "MCPClientConfig",
    "ArgusServer",
    "MCPClient",
    "ToolAdapter",
    "ResourceRegistry",
    # Durable (v1.4.0)
    "DurableConfig",
    "DurableWorkflow",
    "Checkpoint",
    "MemoryCheckpointer",
    "SQLiteCheckpointer",
    "StateManager",
    "idempotent_task",
]


