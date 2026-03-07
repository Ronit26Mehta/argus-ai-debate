# Argus Codebase Symbol Map

> **Root:** `C:\ingester_ops\argus\argus`  
> **Generated:** 2026-03-07 23:39:58  
> **Files scanned:** 164  
> **Total symbols:** 2701

## Summary

| Kind | Count |
|------|------:|
| 🏛️ Class | 427 |
| 🔧 Function | 193 |
| ⚙️ Method | 2081 |

---

## Table of Contents

- [argus/agents/base.py](#argusagentsbasepy)
- [argus/agents/jury.py](#argusagentsjurypy)
- [argus/agents/moderator.py](#argusagentsmoderatorpy)
- [argus/agents/refuter.py](#argusagentsrefuterpy)
- [argus/agents/specialist.py](#argusagentsspecialistpy)
- [argus/cdag/edges.py](#arguscdagedgespy)
- [argus/cdag/graph.py](#arguscdaggraphpy)
- [argus/cdag/nodes.py](#arguscdagnodespy)
- [argus/cdag/propagation.py](#arguscdagpropagationpy)
- [argus/cli.py](#argusclipy)
- [argus/core/config.py](#arguscoreconfigpy)
- [argus/core/context_caching.py](#arguscorecontextcachingpy)
- [argus/core/context_compression.py](#arguscorecontextcompressionpy)
- [argus/core/llm/anthropic.py](#arguscorellmanthropicpy)
- [argus/core/llm/azure_openai.py](#arguscorellmazureopenaipy)
- [argus/core/llm/base.py](#arguscorellmbasepy)
- [argus/core/llm/bedrock.py](#arguscorellmbedrockpy)
- [argus/core/llm/cerebras.py](#arguscorellmcerebraspy)
- [argus/core/llm/cloudflare.py](#arguscorellmcloudflarepy)
- [argus/core/llm/cohere.py](#arguscorellmcoherepy)
- [argus/core/llm/databricks.py](#arguscorellmdatabrickspy)
- [argus/core/llm/deepseek.py](#arguscorellmdeepseekpy)
- [argus/core/llm/fireworks.py](#arguscorellmfireworkspy)
- [argus/core/llm/gemini.py](#arguscorellmgeminipy)
- [argus/core/llm/groq.py](#arguscorellmgroqpy)
- [argus/core/llm/huggingface.py](#arguscorellmhuggingfacepy)
- [argus/core/llm/litellm.py](#arguscorellmlitellmpy)
- [argus/core/llm/llamacpp.py](#arguscorellmllamacpppy)
- [argus/core/llm/mistral.py](#arguscorellmmistralpy)
- [argus/core/llm/nvidia.py](#arguscorellmnvidiapy)
- [argus/core/llm/ollama.py](#arguscorellmollamapy)
- [argus/core/llm/openai.py](#arguscorellmopenaipy)
- [argus/core/llm/perplexity.py](#arguscorellmperplexitypy)
- [argus/core/llm/registry.py](#arguscorellmregistrypy)
- [argus/core/llm/replicate.py](#arguscorellmreplicatepy)
- [argus/core/llm/sambanova.py](#arguscorellmsambanovapy)
- [argus/core/llm/snowflake.py](#arguscorellmsnowflakepy)
- [argus/core/llm/together.py](#arguscorellmtogetherpy)
- [argus/core/llm/vertex.py](#arguscorellmvertexpy)
- [argus/core/llm/vllm.py](#arguscorellmvllmpy)
- [argus/core/llm/watsonx.py](#arguscorellmwatsonxpy)
- [argus/core/llm/xai.py](#arguscorellmxaipy)
- [argus/core/models.py](#arguscoremodelspy)
- [argus/core/openapi.py](#arguscoreopenapipy)
- [argus/crux/agent_card.py](#arguscruxagentcardpy)
- [argus/crux/auction.py](#arguscruxauctionpy)
- [argus/crux/brp.py](#arguscruxbrppy)
- [argus/crux/claim_bundle.py](#arguscruxclaimbundlepy)
- [argus/crux/edr.py](#arguscruxedrpy)
- [argus/crux/ledger.py](#arguscruxledgerpy)
- [argus/crux/models.py](#arguscruxmodelspy)
- [argus/crux/orchestrator.py](#arguscruxorchestratorpy)
- [argus/crux/routing.py](#arguscruxroutingpy)
- [argus/crux/visualization.py](#arguscruxvisualizationpy)
- [argus/debate/visualization.py](#argusdebatevisualizationpy)
- [argus/decision/bayesian.py](#argusdecisionbayesianpy)
- [argus/decision/calibration.py](#argusdecisioncalibrationpy)
- [argus/decision/eig.py](#argusdecisioneigpy)
- [argus/decision/planner.py](#argusdecisionplannerpy)
- [argus/durable/checkpointer.py](#argusdurablecheckpointerpy)
- [argus/durable/config.py](#argusdurableconfigpy)
- [argus/durable/state.py](#argusdurablestatepy)
- [argus/durable/tasks.py](#argusdurabletaskspy)
- [argus/durable/workflow.py](#argusdurableworkflowpy)
- [argus/evaluation/benchmarks/base.py](#argusevaluationbenchmarksbasepy)
- [argus/evaluation/benchmarks/debate_quality.py](#argusevaluationbenchmarksdebatequalitypy)
- [argus/evaluation/benchmarks/evidence_analysis.py](#argusevaluationbenchmarksevidenceanalysispy)
- [argus/evaluation/benchmarks/reasoning_depth.py](#argusevaluationbenchmarksreasoningdepthpy)
- [argus/evaluation/datasets/global_benchmarks.py](#argusevaluationdatasetsglobalbenchmarkspy)
- [argus/evaluation/datasets/loader.py](#argusevaluationdatasetsloaderpy)
- [argus/evaluation/runner/benchmark_runner.py](#argusevaluationrunnerbenchmarkrunnerpy)
- [argus/evaluation/runner/report_generator.py](#argusevaluationrunnerreportgeneratorpy)
- [argus/evaluation/runner/results_aggregator.py](#argusevaluationrunnerresultsaggregatorpy)
- [argus/evaluation/scoring/aggregate.py](#argusevaluationscoringaggregatepy)
- [argus/evaluation/scoring/metrics.py](#argusevaluationscoringmetricspy)
- [argus/evaluation/scoring/standard_metrics.py](#argusevaluationscoringstandardmetricspy)
- [argus/hitl/callbacks.py](#argushitlcallbackspy)
- [argus/hitl/config.py](#argushitlconfigpy)
- [argus/hitl/handlers.py](#argushitlhandlerspy)
- [argus/hitl/middleware.py](#argushitlmiddlewarepy)
- [argus/knowledge/chunking.py](#argusknowledgechunkingpy)
- [argus/knowledge/connectors/arxiv.py](#argusknowledgeconnectorsarxivpy)
- [argus/knowledge/connectors/base.py](#argusknowledgeconnectorsbasepy)
- [argus/knowledge/connectors/crossref.py](#argusknowledgeconnectorscrossrefpy)
- [argus/knowledge/connectors/web.py](#argusknowledgeconnectorswebpy)
- [argus/knowledge/embeddings.py](#argusknowledgeembeddingspy)
- [argus/knowledge/indexing.py](#argusknowledgeindexingpy)
- [argus/knowledge/ingestion.py](#argusknowledgeingestionpy)
- [argus/mcp/client.py](#argusmcpclientpy)
- [argus/mcp/config.py](#argusmcpconfigpy)
- [argus/mcp/resources.py](#argusmcpresourcespy)
- [argus/mcp/server.py](#argusmcpserverpy)
- [argus/mcp/tools.py](#argusmcptoolspy)
- [argus/memory/config.py](#argusmemoryconfigpy)
- [argus/memory/long_term.py](#argusmemorylongtermpy)
- [argus/memory/semantic_cache.py](#argusmemorysemanticcachepy)
- [argus/memory/short_term.py](#argusmemoryshorttermpy)
- [argus/memory/store.py](#argusmemorystorepy)
- [argus/metrics/collector.py](#argusmetricscollectorpy)
- [argus/metrics/traces.py](#argusmetricstracespy)
- [argus/orchestrator.py](#argusorchestratorpy)
- [argus/outputs/plotting.py](#argusoutputsplottingpy)
- [argus/outputs/reports.py](#argusoutputsreportspy)
- [argus/provenance/integrity.py](#argusprovenanceintegritypy)
- [argus/provenance/ledger.py](#argusprovenanceledgerpy)
- [argus/retrieval/cite_critique.py](#argusretrievalcitecritiquepy)
- [argus/retrieval/hybrid.py](#argusretrievalhybridpy)
- [argus/retrieval/reranker.py](#argusretrievalrerankerpy)
- [argus/tools/base.py](#argustoolsbasepy)
- [argus/tools/cache.py](#argustoolscachepy)
- [argus/tools/executor.py](#argustoolsexecutorpy)
- [argus/tools/guardrails.py](#argustoolsguardrailspy)
- [argus/tools/integrations/__init__.py](#argustoolsintegrationsinitpy)
- [argus/tools/integrations/ai_agents/agentmail.py](#argustoolsintegrationsaiagentsagentmailpy)
- [argus/tools/integrations/ai_agents/agentops.py](#argustoolsintegrationsaiagentsagentopspy)
- [argus/tools/integrations/ai_agents/freeplay.py](#argustoolsintegrationsaiagentsfreeplaypy)
- [argus/tools/integrations/ai_agents/goodmem.py](#argustoolsintegrationsaiagentsgoodmempy)
- [argus/tools/integrations/cloud/bigquery.py](#argustoolsintegrationscloudbigquerypy)
- [argus/tools/integrations/cloud/cloud_trace.py](#argustoolsintegrationscloudcloudtracepy)
- [argus/tools/integrations/cloud/pubsub.py](#argustoolsintegrationscloudpubsubpy)
- [argus/tools/integrations/cloud/vertex_ai.py](#argustoolsintegrationscloudvertexaipy)
- [argus/tools/integrations/communication/mailgun.py](#argustoolsintegrationscommunicationmailgunpy)
- [argus/tools/integrations/communication/paypal.py](#argustoolsintegrationscommunicationpaypalpy)
- [argus/tools/integrations/communication/stripe_tool.py](#argustoolsintegrationscommunicationstripetoolpy)
- [argus/tools/integrations/database/pandas_tool.py](#argustoolsintegrationsdatabasepandastoolpy)
- [argus/tools/integrations/database/sql.py](#argustoolsintegrationsdatabasesqlpy)
- [argus/tools/integrations/devops/daytona.py](#argustoolsintegrationsdevopsdaytonapy)
- [argus/tools/integrations/devops/gitlab.py](#argustoolsintegrationsdevopsgitlabpy)
- [argus/tools/integrations/devops/n8n.py](#argustoolsintegrationsdevopsn8npy)
- [argus/tools/integrations/devops/postman.py](#argustoolsintegrationsdevopspostmanpy)
- [argus/tools/integrations/finance/weather.py](#argustoolsintegrationsfinanceweatherpy)
- [argus/tools/integrations/finance/yahoo_finance.py](#argustoolsintegrationsfinanceyahoofinancepy)
- [argus/tools/integrations/media_ai/cartesia.py](#argustoolsintegrationsmediaaicartesiapy)
- [argus/tools/integrations/media_ai/elevenlabs.py](#argustoolsintegrationsmediaaielevenlabspy)
- [argus/tools/integrations/media_ai/huggingface.py](#argustoolsintegrationsmediaaihuggingfacepy)
- [argus/tools/integrations/observability/arize.py](#argustoolsintegrationsobservabilityarizepy)
- [argus/tools/integrations/observability/mlflow_tool.py](#argustoolsintegrationsobservabilitymlflowtoolpy)
- [argus/tools/integrations/observability/monocle.py](#argustoolsintegrationsobservabilitymonoclepy)
- [argus/tools/integrations/observability/phoenix.py](#argustoolsintegrationsobservabilityphoenixpy)
- [argus/tools/integrations/observability/wandb_weave.py](#argustoolsintegrationsobservabilitywandbweavepy)
- [argus/tools/integrations/productivity/filesystem.py](#argustoolsintegrationsproductivityfilesystempy)
- [argus/tools/integrations/productivity/github.py](#argustoolsintegrationsproductivitygithubpy)
- [argus/tools/integrations/productivity/json_tool.py](#argustoolsintegrationsproductivityjsontoolpy)
- [argus/tools/integrations/productivity/python_repl.py](#argustoolsintegrationsproductivitypythonreplpy)
- [argus/tools/integrations/productivity/shell.py](#argustoolsintegrationsproductivityshellpy)
- [argus/tools/integrations/productivity_tools/asana.py](#argustoolsintegrationsproductivitytoolsasanapy)
- [argus/tools/integrations/productivity_tools/atlassian.py](#argustoolsintegrationsproductivitytoolsatlassianpy)
- [argus/tools/integrations/productivity_tools/linear.py](#argustoolsintegrationsproductivitytoolslinearpy)
- [argus/tools/integrations/productivity_tools/notion.py](#argustoolsintegrationsproductivitytoolsnotionpy)
- [argus/tools/integrations/search/arxiv_tool.py](#argustoolsintegrationssearcharxivtoolpy)
- [argus/tools/integrations/search/brave.py](#argustoolsintegrationssearchbravepy)
- [argus/tools/integrations/search/duckduckgo.py](#argustoolsintegrationssearchduckduckgopy)
- [argus/tools/integrations/search/exa.py](#argustoolsintegrationssearchexapy)
- [argus/tools/integrations/search/tavily.py](#argustoolsintegrationssearchtavilypy)
- [argus/tools/integrations/search/wikipedia.py](#argustoolsintegrationssearchwikipediapy)
- [argus/tools/integrations/vectordb/chroma.py](#argustoolsintegrationsvectordbchromapy)
- [argus/tools/integrations/vectordb/mongodb.py](#argustoolsintegrationsvectordbmongodbpy)
- [argus/tools/integrations/vectordb/pinecone_tool.py](#argustoolsintegrationsvectordbpineconetoolpy)
- [argus/tools/integrations/vectordb/qdrant.py](#argustoolsintegrationsvectordbqdrantpy)
- [argus/tools/integrations/web/jina_reader.py](#argustoolsintegrationswebjinareaderpy)
- [argus/tools/integrations/web/requests_tool.py](#argustoolsintegrationswebrequeststoolpy)
- [argus/tools/integrations/web/scraper.py](#argustoolsintegrationswebscraperpy)
- [argus/tools/integrations/web/youtube.py](#argustoolsintegrationswebyoutubepy)
- [argus/tools/registry.py](#argustoolsregistrypy)

---

## `argus/agents/base.py`

### 🏛️ `class AgentRole(str, Enum)`  <sub>line 26</sub>

> Agent roles in the debate system.  

### 🏛️ `class AgentConfig(BaseModel)`  <sub>line 35</sub>

> Base configuration for agents.  

### 🏛️ `class AgentResponse()`  <sub>line 76</sub>

> Decorators: `@dataclass`  
> Response from an agent action.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `failed` | `(self) -> bool` | 96 | Check if response failed. |

### 🏛️ `class BaseAgent(ABC)`  <sub>line 101</sub>

> Abstract base class for all agents.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, llm: 'BaseLLM', config: Optional[AgentConfig] = None, name: Optional[str] = None)` | 113 | Initialize agent. |
| ⚙️ `@property` | `name` | `(self) -> str` | 141 | Get agent name. |
| ⚙️ `@property` | `role` | `(self) -> AgentRole` | 146 | Get agent role. |
| ⚙️ `@abstractmethod` | `get_system_prompt` | `(self) -> str` | 151 | Get the agent's system prompt. |
| ⚙️ `@abstractmethod` | `act` | `(self, graph: 'CDAG', context: dict[str, Any]) -> AgentResponse` | 161 | Perform the agent's main action. |
| ⚙️ | `generate` | `(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str` | 178 | Generate a response from the LLM. |
| ⚙️ | `log_action` | `(self, action: str, details: dict[str, Any]) -> None` | 209 | Log an action to history. |
| ⚙️ | `get_history` | `(self) -> list[dict[str, Any]]` | 230 | Get action history. |
| ⚙️ | `__repr__` | `(self) -> str` | 234 | String representation. |

---

## `argus/agents/jury.py`

### 🏛️ `class JuryConfig(AgentConfig)`  <sub>line 32</sub>

> Configuration for Jury agent.  

### 🏛️ `class Verdict()`  <sub>line 52</sub>

> Decorators: `@dataclass`  
> Verdict on a proposition.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 77 | Convert to dictionary. |

### 🏛️ `class Jury(BaseAgent)`  <sub>line 92</sub>

> Jury agent for aggregating evidence and rendering verdicts.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, llm: 'BaseLLM', config: Optional[JuryConfig] = None)` | 108 | Initialize Jury. |
| ⚙️ | `get_system_prompt` | `(self) -> str` | 122 | Get Jury system prompt. |
| ⚙️ | `act` | `(self, graph: 'CDAG', context: dict[str, Any]) -> AgentResponse` | 155 | Perform jury action. |
| ⚙️ | `evaluate` | `(self, graph: 'CDAG', proposition_id: str) -> Verdict` | 203 | Evaluate a proposition and render verdict. |
| ⚙️ | `evaluate_all` | `(self, graph: 'CDAG') -> list[Verdict]` | 284 | Evaluate all propositions in the graph. |
| ⚙️ | `_generate_reasoning` | `(self, prop: Any, posterior: float, supporting: list[Any], attacking: list[Any], label: str) -> str` | 306 | Generate LLM reasoning for verdict. |
| ⚙️ | `compute_disagreement` | `(self, graph: 'CDAG', proposition_id: str) -> float` | 335 | Compute disagreement index for a proposition. |

---

## `argus/agents/moderator.py`

### 🏛️ `class ModeratorConfig(AgentConfig)`  <sub>line 35</sub>

> Configuration for Moderator agent.  

### 🏛️ `class DebateAgenda()`  <sub>line 62</sub>

> Decorators: `@dataclass`  
> Agenda for a debate session.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 85 | Convert to dictionary. |

### 🏛️ `class Moderator(BaseAgent)`  <sub>line 99</sub>

> Moderator agent for coordinating debates.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, llm: 'BaseLLM', config: Optional[ModeratorConfig] = None)` | 117 | Initialize Moderator. |
| ⚙️ | `get_system_prompt` | `(self) -> str` | 132 | Get Moderator system prompt. |
| ⚙️ | `act` | `(self, graph: 'CDAG', context: dict[str, Any]) -> AgentResponse` | 152 | Perform moderator action. |
| ⚙️ | `create_agenda` | `(self, graph: 'CDAG', proposition_id: str) -> DebateAgenda` | 201 | Create a debate agenda for a proposition. |
| ⚙️ | `advance_round` | `(self, graph: 'CDAG', agenda: Optional[DebateAgenda] = None) -> dict[str, Any]` | 281 | Advance to the next debate round. |
| ⚙️ | `should_stop` | `(self, graph: 'CDAG', agenda: Optional[DebateAgenda] = None) -> tuple[bool, str]` | 319 | Check if debate should stop. |
| ⚙️ | `_evaluate_round` | `(self, graph: 'CDAG', context: dict[str, Any]) -> AgentResponse` | 357 | Evaluate the current round. |

---

## `argus/agents/refuter.py`

### 🏛️ `class RefuterConfig(AgentConfig)`  <sub>line 33</sub>

> Configuration for Refuter agent.  

### 🏛️ `class Refuter(BaseAgent)`  <sub>line 53</sub>

> Refuter agent for finding counter-evidence.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, llm: 'BaseLLM', config: Optional[RefuterConfig] = None)` | 70 | Initialize Refuter. |
| ⚙️ | `get_system_prompt` | `(self) -> str` | 84 | Get Refuter system prompt. |
| ⚙️ | `act` | `(self, graph: 'CDAG', context: dict[str, Any]) -> AgentResponse` | 122 | Perform refuter action. |
| ⚙️ | `generate_rebuttals` | `(self, graph: 'CDAG', proposition_id: str) -> list[Rebuttal]` | 170 | Generate rebuttals for evidence on a proposition. |
| ⚙️ | `find_contradictions` | `(self, graph: 'CDAG', proposition_id: str) -> list[dict[str, Any]]` | 273 | Find contradictions between evidence items. |
| ⚙️ `@staticmethod` | `_strip_json` | `(text: str) -> str` | 335 | Strip markdown code fences from LLM JSON responses. |

---

## `argus/agents/specialist.py`

### 🏛️ `class SpecialistConfig(AgentConfig)`  <sub>line 34</sub>

> Configuration for Specialist agent.  

### 🏛️ `class Specialist(BaseAgent)`  <sub>line 59</sub>

> Specialist agent for domain-specific evidence gathering.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, llm: 'BaseLLM', config: Optional[SpecialistConfig] = None, domain: Optional[str] = None)` | 76 | Initialize Specialist. |
| ⚙️ `@property` | `domain` | `(self) -> str` | 96 | Get domain of expertise. |
| ⚙️ | `get_system_prompt` | `(self) -> str` | 100 | Get Specialist system prompt. |
| ⚙️ | `act` | `(self, graph: 'CDAG', context: dict[str, Any]) -> AgentResponse` | 128 | Perform specialist action. |
| ⚙️ | `gather_evidence` | `(self, graph: 'CDAG', proposition_id: str, index: Optional['HybridIndex'] = None, embedding_generator: Any = None) -> list[Evidence]` | 182 | Gather evidence for a proposition. |
| ⚙️ | `evaluate_chunk` | `(self, graph: 'CDAG', proposition_id: str, chunk_text: str) -> dict[str, Any]` | 286 | Evaluate a chunk for evidence. |
| ⚙️ | `_create_evidence` | `(self, evaluation: dict[str, Any], chunk: Any, retrieval_score: float) -> Evidence` | 350 | Create Evidence node from evaluation. |

---

## `argus/cdag/edges.py`

### 🏛️ `class EdgeType(str, Enum)`  <sub>line 27</sub>

> Type of edge in the C-DAG.  

### 🏛️ `class EdgePolarity(str, Enum)`  <sub>line 39</sub>

> Polarity of an edge (positive/negative influence).  

### 🏛️ `class Edge(BaseModel)`  <sub>line 59</sub>

> An edge connecting two nodes in the C-DAG.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `model_post_init` | `(self, __context: Any) -> None` | 156 | Set default polarity based on edge type. |
| ⚙️ `@computed_field`, `@property` | `effective_weight` | `(self) -> float` | 166 | Compute effective weight for propagation. |
| ⚙️ `@computed_field`, `@property` | `signed_weight` | `(self) -> float` | 176 | Compute signed weight for influence propagation. |
| ⚙️ `@computed_field`, `@property` | `is_supporting` | `(self) -> bool` | 189 | Check if this is a supporting edge. |
| ⚙️ `@computed_field`, `@property` | `is_attacking` | `(self) -> bool` | 195 | Check if this is an attacking edge. |
| ⚙️ | `update_weight` | `(self, weight: Optional[float] = None, confidence: Optional[float] = None, relevance: Optional[float] = None, quality: Optional[float] = None) -> None` | 199 | Update edge weights. |
| ⚙️ | `__hash__` | `(self) -> int` | 225 | Hash by edge ID. |
| ⚙️ | `__eq__` | `(self, other: object) -> bool` | 229 | Equality based on edge ID. |
| ⚙️ | `to_tuple` | `(self) -> tuple[str, str]` | 235 | Return (source, target) tuple for graph operations. |
| ⚙️ | `to_prov_dict` | `(self) -> dict[str, Any]` | 239 | Convert edge to PROV-O compatible dictionary. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_edge_id` | `() -> str` | 22 | Generate a unique edge identifier. |
| 🔧 | `create_support_edge` | `(source_id: str, target_id: str, confidence: float = 1.0, relevance: float = 1.0, **kwargs: Any) -> Edge` | 258 | Create a support edge. |
| 🔧 | `create_attack_edge` | `(source_id: str, target_id: str, confidence: float = 1.0, relevance: float = 1.0, **kwargs: Any) -> Edge` | 289 | Create an attack edge. |
| 🔧 | `create_rebuttal_edge` | `(source_id: str, target_id: str, confidence: float = 1.0, **kwargs: Any) -> Edge` | 320 | Create a rebuttal edge. |

---

## `argus/cdag/graph.py`

### 🏛️ `class CDAG()`  <sub>line 38</sub>

> Conceptual Debate Graph.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None)` | 71 | Initialize a new C-DAG. |
| ⚙️ | `add_node` | `(self, node: NodeBase) -> str` | 107 | Add a node to the graph. |
| ⚙️ | `add_proposition` | `(self, prop: Proposition) -> str` | 134 | Add a proposition to the graph. |
| ⚙️ | `add_evidence` | `(self, evidence: Evidence, target_id: str, edge_type: EdgeType = EdgeType.SUPPORTS, **edge_kwargs: Any) -> tuple[str, str]` | 146 | Add evidence linked to a target node. |
| ⚙️ | `add_rebuttal` | `(self, rebuttal: Rebuttal, target_id: str, **edge_kwargs: Any) -> tuple[str, str]` | 198 | Add a rebuttal attacking another node. |
| ⚙️ | `get_node` | `(self, node_id: str) -> Optional[NodeBase]` | 239 | Get a node by ID. |
| ⚙️ | `get_proposition` | `(self, prop_id: str) -> Optional[Proposition]` | 251 | Get a proposition by ID. |
| ⚙️ | `remove_node` | `(self, node_id: str) -> bool` | 266 | Remove a node and its connected edges. |
| ⚙️ | `add_edge` | `(self, edge: Edge) -> str` | 299 | Add an edge to the graph. |
| ⚙️ | `get_edge` | `(self, edge_id: str) -> Optional[Edge]` | 329 | Get an edge by ID. |
| ⚙️ | `remove_edge` | `(self, edge_id: str) -> bool` | 341 | Remove an edge. |
| ⚙️ | `get_incoming_edges` | `(self, node_id: str, edge_type: Optional[EdgeType] = None, polarity: Optional[EdgePolarity] = None) -> list[Edge]` | 373 | Get edges pointing to a node. |
| ⚙️ | `get_outgoing_edges` | `(self, node_id: str, edge_type: Optional[EdgeType] = None) -> list[Edge]` | 400 | Get edges originating from a node. |
| ⚙️ | `get_supporting_evidence` | `(self, prop_id: str) -> list[Evidence]` | 423 | Get all evidence supporting a proposition. |
| ⚙️ | `get_attacking_evidence` | `(self, prop_id: str) -> list[Evidence]` | 447 | Get all evidence attacking a proposition. |
| ⚙️ | `get_all_propositions` | `(self) -> list[Proposition]` | 471 | Get all propositions in the graph. |
| ⚙️ | `get_all_evidence` | `(self) -> list[Evidence]` | 484 | Get all evidence nodes in the graph. |
| ⚙️ | `get_support_path` | `(self, prop_id: str, max_depth: int = 3) -> list[list[str]]` | 499 | Get paths of support to a proposition. |
| ⚙️ | `_trace_path` | `(current_id: str, current_path: list[str], depth: int) -> None` | 516 |  |
| ⚙️ | `get_attack_path` | `(self, prop_id: str, max_depth: int = 3) -> list[list[str]]` | 537 | Get paths of attack to a proposition. |
| ⚙️ | `_trace_path` | `(current_id: str, current_path: list[str], depth: int) -> None` | 554 |  |
| ⚙️ | `compute_support_score` | `(self, prop_id: str) -> float` | 577 | Compute aggregate support score for a proposition. |
| ⚙️ | `compute_attack_score` | `(self, prop_id: str) -> float` | 604 | Compute aggregate attack score for a proposition. |
| ⚙️ | `compute_net_influence` | `(self, prop_id: str) -> float` | 630 | Compute net influence (support - attack). |
| ⚙️ `@property` | `num_nodes` | `(self) -> int` | 645 | Total number of nodes. |
| ⚙️ `@property` | `num_edges` | `(self) -> int` | 650 | Total number of edges. |
| ⚙️ `@property` | `num_propositions` | `(self) -> int` | 655 | Number of propositions. |
| ⚙️ `@property` | `num_evidence` | `(self) -> int` | 660 | Number of evidence nodes. |
| ⚙️ | `__repr__` | `(self) -> str` | 664 | String representation. |
| ⚙️ | `summary` | `(self) -> dict[str, Any]` | 673 | Get graph summary statistics. |
| ⚙️ | `to_networkx` | `(self) -> nx.DiGraph` | 690 | Get underlying NetworkX graph. |
| ⚙️ | `clear` | `(self) -> None` | 699 | Clear all nodes and edges from the graph. |

---

## `argus/cdag/nodes.py`

### 🏛️ `class NodeStatus(str, Enum)`  <sub>line 27</sub>

> Status of a node in the debate.  

### 🏛️ `class NodeBase(BaseModel)`  <sub>line 38</sub>

> Base class for all C-DAG nodes.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `score` | `(self) -> float` | 104 | Get the computed influence score. |
| ⚙️ `@score.setter` | `score` | `(self, value: float) -> None` | 109 | Set the influence score. |
| ⚙️ `@field_validator('confidence')`, `@classmethod` | `validate_confidence` | `(cls, v: float) -> float` | 115 | Ensure confidence is bounded. |
| ⚙️ | `update_status` | `(self, new_status: NodeStatus) -> None` | 119 | Update node status with timestamp. |
| ⚙️ | `__hash__` | `(self) -> int` | 129 | Hash by node ID. |
| ⚙️ | `__eq__` | `(self, other: object) -> bool` | 133 | Equality based on node ID. |

### 🏛️ `class Proposition(NodeBase)`  <sub>line 140</sub>

> A proposition (hypothesis) to be evaluated.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `log_odds_prior` | `(self) -> float` | 192 | Prior in log-odds space. |
| ⚙️ `@computed_field`, `@property` | `log_odds_posterior` | `(self) -> float` | 200 | Posterior in log-odds space. |
| ⚙️ | `update_posterior` | `(self, new_posterior: float) -> None` | 206 | Update the posterior probability. |

### 🏛️ `class EvidenceType(str, Enum)`  <sub>line 230</sub>

> Type of evidence.  

### 🏛️ `class Evidence(NodeBase)`  <sub>line 239</sub>

> Evidence node supporting or attacking a proposition.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `effective_weight` | `(self) -> float` | 305 | Compute effective weight for scoring. |
| ⚙️ `@computed_field`, `@property` | `is_supporting` | `(self) -> bool` | 315 | Check if this is supporting evidence. |
| ⚙️ `@computed_field`, `@property` | `is_attacking` | `(self) -> bool` | 321 | Check if this is attacking evidence. |
| ⚙️ | `add_citation` | `(self, doc_id: str, chunk_id: str, quote: Optional[str] = None, url: Optional[str] = None) -> None` | 325 | Add a citation to this evidence. |

### 🏛️ `class Rebuttal(NodeBase)`  <sub>line 352</sub>

> A rebuttal or counter-argument.  

### 🏛️ `class Finding(NodeBase)`  <sub>line 387</sub>

> A factual finding or observation.  

### 🏛️ `class Assumption(NodeBase)`  <sub>line 421</sub>

> An assumption or prior belief.  

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_node_id` | `(prefix: str) -> str` | 22 | Generate a unique node identifier with prefix. |

---

## `argus/cdag/propagation.py`

### 🏛️ `class PropagationConfig()`  <sub>line 34</sub>

> Decorators: `@dataclass`  
> Configuration for influence propagation.  

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `sigmoid` | `(x: float) -> float` | 56 | Sigmoid squashing function. |
| 🔧 | `squash` | `(x: float, method: str = 'tanh', temperature: float = 1.0) -> float` | 71 | Apply squashing function to a value. |
| 🔧 | `log_odds` | `(p: float) -> float` | 99 | Convert probability to log-odds. |
| 🔧 | `from_log_odds` | `(lo: float) -> float` | 113 | Convert log-odds to probability. |
| 🔧 | `compute_log_likelihood_ratio` | `(confidence: float, temperature: float = 1.0) -> float` | 126 | Compute log-likelihood ratio from confidence. |
| 🔧 | `propagate_influence` | `(graph: 'CDAG', config: Optional[PropagationConfig] = None) -> dict[str, float]` | 159 | Propagate influence through the C-DAG. |
| 🔧 | `compute_posterior` | `(graph: 'CDAG', prop_id: str, config: Optional[PropagationConfig] = None) -> float` | 277 | Compute posterior probability for a proposition. |
| 🔧 | `compute_all_posteriors` | `(graph: 'CDAG', config: Optional[PropagationConfig] = None) -> dict[str, float]` | 359 | Compute posteriors for all propositions. |
| 🔧 | `compute_disagreement_index` | `(graph: 'CDAG', prop_id: str) -> float` | 382 | Compute disagreement index for a proposition. |
| 🔧 | `get_counter_evidence` | `(graph: 'CDAG', prop_id: str, threshold: float = 0.3) -> list[dict[str, Any]]` | 413 | Get evidence that could flip the verdict. |

---

## `argus/cli.py`

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `setup_parser` | `() -> argparse.ArgumentParser` | 29 | Setup argument parser. |
| 🔧 | `cmd_debate` | `(args: argparse.Namespace) -> int` | 432 | Run debate command. |
| 🔧 | `cmd_evaluate` | `(args: argparse.Namespace) -> int` | 476 | Run quick evaluation command. |
| 🔧 | `cmd_ingest` | `(args: argparse.Namespace) -> int` | 507 | Run ingest command. |
| 🔧 | `cmd_providers` | `(args: argparse.Namespace) -> int` | 557 | List available LLM providers. |
| 🔧 | `cmd_benchmark` | `(args: argparse.Namespace) -> int` | 597 | Run evaluation benchmarks. |
| 🔧 | `cmd_datasets` | `(args: argparse.Namespace) -> int` | 656 | List available evaluation datasets. |
| 🔧 | `cmd_report` | `(args: argparse.Namespace) -> int` | 696 | Generate report from debate results. |
| 🔧 | `cmd_tools` | `(args: argparse.Namespace) -> int` | 746 | List registered tools. |
| 🔧 | `cmd_embeddings` | `(args: argparse.Namespace) -> int` | 799 | List available embedding providers. |
| 🔧 | `cmd_score` | `(args: argparse.Namespace) -> int` | 883 | Compute ARGUS scoring metrics. |
| 🔧 | `cmd_config` | `(args: argparse.Namespace) -> int` | 927 | Show configuration. |
| 🔧 | `cmd_openapi` | `(args: argparse.Namespace) -> int` | 949 | Generate tools from OpenAPI specifications. |
| 🔧 | `cmd_cache` | `(args: argparse.Namespace) -> int` | 1007 | Manage context caching. |
| 🔧 | `cmd_compress` | `(args: argparse.Namespace) -> int` | 1065 | Compress context or conversation history. |
| 🔧 | `cmd_visualize` | `(args: argparse.Namespace) -> int` | 1135 | Generate debate visualizations. |
| 🔧 | `main` | `() -> int` | 1215 | Main CLI entry point. |

---

## `argus/core/config.py`

### 🏛️ `class LLMProviderConfig(BaseSettings)`  <sub>line 50</sub>

> Configuration for LLM providers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `has_provider` | `(self, provider: str) -> bool` | 103 | Check if a provider is configured with valid credentials. |
| ⚙️ | `get_available_providers` | `(self) -> list[str]` | 140 | Get list of providers with valid credentials. |

### 🏛️ `class EmbeddingProviderConfig(BaseSettings)`  <sub>line 151</sub>

> Configuration for embedding providers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `has_provider` | `(self, provider: str) -> bool` | 199 | Check if an embedding provider is configured. |
| ⚙️ | `get_available_providers` | `(self) -> list[str]` | 224 | Get list of configured embedding providers. |

### 🏛️ `class RetrievalConfig(BaseSettings)`  <sub>line 235</sub>

> Configuration for document retrieval and evidence engineering.  

### 🏛️ `class ChunkingConfig(BaseSettings)`  <sub>line 290</sub>

> Configuration for document chunking.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@field_validator('chunk_overlap')`, `@classmethod` | `validate_overlap` | `(cls, v: int, info: Any) -> int` | 339 | Ensure overlap is less than chunk size. |
| ⚙️ `@model_validator(mode='after')` | `validate_chunk_params` | `(self) -> 'ChunkingConfig'` | 346 | Validate chunk parameters are consistent. |

### 🏛️ `class CalibrationConfig(BaseSettings)`  <sub>line 361</sub>

> Configuration for confidence calibration.  

### 🏛️ `class ProvenanceConfig(BaseSettings)`  <sub>line 399</sub>

> Configuration for provenance tracking.  

### 🏛️ `class ArgusConfig(BaseSettings)`  <sub>line 439</sub>

> Main ARGUS configuration.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@model_validator(mode='after')` | `validate_config` | `(self) -> 'ArgusConfig'` | 536 | Validate overall configuration consistency. |
| ⚙️ | `configure_logging` | `(self) -> None` | 546 | Configure logging based on settings. |
| ⚙️ | `get_model_for_provider` | `(self, provider: Optional[str] = None) -> str` | 554 | Get appropriate model name for a provider. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 `@lru_cache(maxsize=1)` | `get_config` | `() -> ArgusConfig` | 586 | Get the global ARGUS configuration. |
| 🔧 | `reset_config` | `() -> None` | 608 | Reset the global configuration. |
| 🔧 | `set_config` | `(config: ArgusConfig) -> None` | 620 | Set the global configuration explicitly. |

---

## `argus/core/context_caching.py`

### 🏛️ `class CacheEntry(Generic[T])`  <sub>line 30</sub>

> Decorators: `@dataclass`  
> A cached entry with metadata.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `is_expired` | `(self) -> bool` | 42 | Check if entry has expired. |
| ⚙️ `@property` | `age_seconds` | `(self) -> float` | 49 | Get entry age in seconds. |
| ⚙️ | `touch` | `(self)` | 53 | Update access time and count. |

### 🏛️ `class CacheStats()`  <sub>line 60</sub>

> Decorators: `@dataclass`  
> Cache statistics.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `hit_rate` | `(self) -> float` | 72 | Calculate hit rate. |

### 🏛️ `class CacheBackend(ABC)`  <sub>line 78</sub>

> Abstract base class for cache backends.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@abstractmethod` | `get` | `(self, key: str) -> Optional[CacheEntry]` | 82 | Get entry by key. |
| ⚙️ `@abstractmethod` | `set` | `(self, entry: CacheEntry) -> None` | 87 | Set entry. |
| ⚙️ `@abstractmethod` | `delete` | `(self, key: str) -> bool` | 92 | Delete entry by key. |
| ⚙️ `@abstractmethod` | `exists` | `(self, key: str) -> bool` | 97 | Check if key exists. |
| ⚙️ `@abstractmethod` | `clear` | `(self) -> int` | 102 | Clear all entries. Returns count of cleared entries. |
| ⚙️ `@abstractmethod` | `keys` | `(self) -> List[str]` | 107 | Get all keys. |
| ⚙️ `@abstractmethod` | `size` | `(self) -> int` | 112 | Get total number of entries. |

### 🏛️ `class MemoryBackend(CacheBackend)`  <sub>line 117</sub>

> In-memory cache backend with LRU eviction.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_entries: int = 1000, max_size_bytes: int = 100 * 1024 * 1024)` | 120 |  |
| ⚙️ | `get` | `(self, key: str) -> Optional[CacheEntry]` | 127 |  |
| ⚙️ | `set` | `(self, entry: CacheEntry) -> None` | 136 |  |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 157 |  |
| ⚙️ | `exists` | `(self, key: str) -> bool` | 165 |  |
| ⚙️ | `clear` | `(self) -> int` | 168 |  |
| ⚙️ | `keys` | `(self) -> List[str]` | 175 |  |
| ⚙️ | `size` | `(self) -> int` | 178 |  |

### 🏛️ `class FileBackend(CacheBackend)`  <sub>line 182</sub>

> File-based cache backend.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, cache_dir: str = '.argus_cache')` | 185 |  |
| ⚙️ | `_load_index` | `(self)` | 193 | Load cache index from file. |
| ⚙️ | `_save_index` | `(self)` | 202 | Save cache index to file. |
| ⚙️ | `_get_entry_path` | `(self, key: str) -> Path` | 207 | Get file path for entry. |
| ⚙️ | `get` | `(self, key: str) -> Optional[CacheEntry]` | 213 |  |
| ⚙️ | `set` | `(self, entry: CacheEntry) -> None` | 238 |  |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 257 |  |
| ⚙️ | `exists` | `(self, key: str) -> bool` | 270 |  |
| ⚙️ | `clear` | `(self) -> int` | 273 |  |
| ⚙️ | `keys` | `(self) -> List[str]` | 286 |  |
| ⚙️ | `size` | `(self) -> int` | 289 |  |

### 🏛️ `class RedisBackend(CacheBackend)`  <sub>line 293</sub>

> Redis cache backend.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: Optional[str] = None, prefix: str = 'argus:')` | 296 |  |
| ⚙️ | `_make_key` | `(self, key: str) -> str` | 318 |  |
| ⚙️ | `get` | `(self, key: str) -> Optional[CacheEntry]` | 321 |  |
| ⚙️ | `set` | `(self, entry: CacheEntry) -> None` | 339 |  |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 350 |  |
| ⚙️ | `exists` | `(self, key: str) -> bool` | 353 |  |
| ⚙️ | `clear` | `(self) -> int` | 356 |  |
| ⚙️ | `keys` | `(self) -> List[str]` | 362 |  |
| ⚙️ | `size` | `(self) -> int` | 366 |  |

### 🏛️ `class ContextCache()`  <sub>line 370</sub>

> Main context caching interface.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, backend: Optional[CacheBackend] = None, default_ttl: Optional[int] = None, namespace: str = '')` | 387 |  |
| ⚙️ | `_make_key` | `(self, key: str) -> str` | 402 | Create namespaced key. |
| ⚙️ | `_estimate_size` | `(self, value: Any) -> int` | 408 | Estimate size of value in bytes. |
| ⚙️ | `get` | `(self, key: str, default: Any = None) -> Any` | 415 | Get cached value. |
| ⚙️ | `set` | `(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None` | 446 | Set cached value. |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 482 | Delete cached value. |
| ⚙️ | `exists` | `(self, key: str) -> bool` | 487 | Check if key exists and is not expired. |
| ⚙️ | `clear` | `(self) -> int` | 501 | Clear all cached entries. |
| ⚙️ | `get_stats` | `(self) -> CacheStats` | 510 | Get cache statistics. |
| ⚙️ | `cached` | `(self, ttl: Optional[int] = None, key_func: Optional[Callable[..., str]] = None) -> Callable` | 522 | Decorator for caching function results. |
| ⚙️ | `decorator` | `(func: Callable) -> Callable` | 539 |  |
| ⚙️ `@wraps(func)` | `wrapper` | `(*args, **kwargs)` | 541 |  |
| ⚙️ | `get_or_set` | `(self, key: str, factory: Callable[[], T], ttl: Optional[int] = None) -> T` | 566 | Get cached value or compute and cache if missing. |

### 🏛️ `class ConversationCache(ContextCache)`  <sub>line 592</sub>

> Specialized cache for conversation/chat history.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, backend: Optional[CacheBackend] = None, max_messages_per_conversation: int = 100, default_ttl: int = 3600)` | 600 |  |
| ⚙️ | `get_messages` | `(self, conversation_id: str) -> List[Dict[str, Any]]` | 609 | Get all messages for a conversation. |
| ⚙️ | `add_message` | `(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None` | 613 | Add a message to a conversation. |
| ⚙️ | `add_user_message` | `(self, conversation_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None` | 638 | Add a user message. |
| ⚙️ | `add_assistant_message` | `(self, conversation_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None` | 647 | Add an assistant message. |
| ⚙️ | `add_system_message` | `(self, conversation_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None` | 656 | Add a system message. |
| ⚙️ | `get_recent_messages` | `(self, conversation_id: str, count: int = 10) -> List[Dict[str, Any]]` | 665 | Get the most recent messages. |
| ⚙️ | `clear_conversation` | `(self, conversation_id: str) -> bool` | 674 | Clear all messages from a conversation. |
| ⚙️ | `summarize_and_truncate` | `(self, conversation_id: str, summary: str, keep_recent: int = 5) -> None` | 678 | Replace old messages with a summary. |

### 🏛️ `class EmbeddingCache(ContextCache)`  <sub>line 711</sub>

> Specialized cache for embeddings.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, backend: Optional[CacheBackend] = None, default_ttl: int = 86400)` | 719 |  |
| ⚙️ | `_make_embedding_key` | `(self, text: str, model: str) -> str` | 726 | Create cache key from text and model. |
| ⚙️ | `get_embedding` | `(self, text: str, model: str) -> Optional[List[float]]` | 731 | Get cached embedding. |
| ⚙️ | `set_embedding` | `(self, text: str, model: str, embedding: List[float]) -> None` | 740 | Cache an embedding. |
| ⚙️ | `get_or_compute` | `(self, text: str, model: str, compute_func: Callable[[str], List[float]]) -> List[float]` | 750 | Get cached embedding or compute if not cached. |
| ⚙️ | `batch_get_or_compute` | `(self, texts: List[str], model: str, compute_func: Callable[[List[str]], List[List[float]]]) -> List[List[float]]` | 767 | Get cached embeddings or compute missing ones in batch. |

### 🏛️ `class LLMResponseCache(ContextCache)`  <sub>line 802</sub>

> Specialized cache for LLM responses.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, backend: Optional[CacheBackend] = None, default_ttl: int = 3600)` | 810 |  |
| ⚙️ | `_make_response_key` | `(self, prompt: str, model: str, temperature: float, **kwargs) -> str` | 817 | Create cache key from prompt and parameters. |
| ⚙️ | `get_response` | `(self, prompt: str, model: str, temperature: float = 0.0, **kwargs) -> Optional[Dict[str, Any]]` | 844 | Get cached LLM response. |
| ⚙️ | `set_response` | `(self, prompt: str, model: str, response: Dict[str, Any], temperature: float = 0.0, **kwargs) -> None` | 859 | Cache an LLM response. |
| ⚙️ | `cached_completion` | `(self, model: str, temperature: float = 0.0) -> Callable` | 875 | Decorator for caching LLM completion calls. |
| ⚙️ | `decorator` | `(func: Callable) -> Callable` | 888 |  |
| ⚙️ `@wraps(func)` | `wrapper` | `(prompt: str, **kwargs)` | 890 |  |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `create_cache` | `(backend_type: str = 'memory', **kwargs) -> ContextCache` | 923 | Create a context cache with specified backend. |

---

## `argus/core/context_compression.py`

### 🏛️ `class CompressionLevel(Enum)`  <sub>line 21</sub>

> Compression intensity levels.  

### 🏛️ `class CompressionResult()`  <sub>line 30</sub>

> Decorators: `@dataclass`  
> Result of a compression operation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `tokens_saved` | `(self) -> int` | 41 | Number of tokens saved. |
| ⚙️ `@property` | `savings_percentage` | `(self) -> float` | 46 | Percentage of tokens saved. |

### 🏛️ `class Message()`  <sub>line 54</sub>

> Decorators: `@dataclass`  
> A chat message.  

### 🏛️ `class TokenCounter()`  <sub>line 62</sub>

> Token counting utilities.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'gpt-4')` | 65 |  |
| ⚙️ | `_get_encoder` | `(self)` | 69 | Get tiktoken encoder lazily. |
| ⚙️ | `count` | `(self, text: str) -> int` | 82 | Count tokens in text. |
| ⚙️ | `count_messages` | `(self, messages: List[Message]) -> int` | 91 | Count tokens in messages. |

### 🏛️ `class Compressor(ABC)`  <sub>line 104</sub>

> Abstract base class for compression techniques.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@abstractmethod` | `compress` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, **kwargs) -> str` | 110 | Compress text. |
| ⚙️ | `compress_with_result` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, token_counter: Optional[TokenCounter] = None, **kwargs) -> CompressionResult` | 119 | Compress text and return detailed result. |

### 🏛️ `class WhitespaceCompressor(Compressor)`  <sub>line 145</sub>

> Compress by normalizing whitespace.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `compress` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, **kwargs) -> str` | 150 | Normalize whitespace in text. |

### 🏛️ `class PunctuationCompressor(Compressor)`  <sub>line 175</sub>

> Compress by simplifying punctuation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `compress` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, **kwargs) -> str` | 180 | Simplify punctuation. |

### 🏛️ `class StopwordCompressor(Compressor)`  <sub>line 202</sub>

> Compress by removing or abbreviating common words.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `compress` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, **kwargs) -> str` | 261 | Remove stopwords and apply abbreviations. |

### 🏛️ `class SentenceCompressor(Compressor)`  <sub>line 298</sub>

> Compress by simplifying sentences.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `compress` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, **kwargs) -> str` | 342 | Simplify sentences. |

### 🏛️ `class CodeCompressor(Compressor)`  <sub>line 403</sub>

> Compress code blocks and technical content.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `compress` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, **kwargs) -> str` | 408 | Compress code-related content. |

### 🏛️ `class SemanticCompressor(Compressor)`  <sub>line 453</sub>

> Semantic compression using LLM summarization.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, llm_func: Optional[Callable[[str], str]] = None, max_output_tokens: int = 500)` | 463 |  |
| ⚙️ | `compress` | `(self, text: str, level: CompressionLevel = CompressionLevel.MODERATE, **kwargs) -> str` | 471 | Compress using LLM summarization. |

### 🏛️ `class MessageCompressor()`  <sub>line 505</sub>

> Compress conversation message history.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, token_counter: Optional[TokenCounter] = None, compressors: Optional[List[Compressor]] = None, summarizer: Optional[Callable[[str], str]] = None)` | 513 |  |
| ⚙️ | `compress_message` | `(self, message: Message, level: CompressionLevel = CompressionLevel.MODERATE) -> Message` | 527 | Compress a single message. |
| ⚙️ | `compress_messages` | `(self, messages: List[Message], max_tokens: int, preserve_recent: int = 5, preserve_system: bool = True) -> List[Message]` | 545 | Compress message history to fit within token limit. |
| ⚙️ | `sliding_window_compress` | `(self, messages: List[Message], window_size: int, summarize_dropped: bool = True) -> Tuple[List[Message], Optional[str]]` | 623 | Keep most recent messages within window, optionally summarizing dropped ones. |

### 🏛️ `class ContextCompressor()`  <sub>line 656</sub>

> Main context compression interface.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, token_counter: Optional[TokenCounter] = None, llm_summarizer: Optional[Callable[[str], str]] = None)` | 664 |  |
| ⚙️ | `compress_text` | `(self, text: str, target_tokens: Optional[int] = None, level: Optional[CompressionLevel] = None, techniques: Optional[List[str]] = None) -> CompressionResult` | 690 | Compress text. |
| ⚙️ | `compress_messages` | `(self, messages: List[Message], max_tokens: int, **kwargs) -> List[Message]` | 769 | Compress conversation messages. |
| ⚙️ | `auto_compress` | `(self, content: Any, target_tokens: int) -> Any` | 780 | Auto-detect content type and compress appropriately. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `compress_text` | `(text: str, level: CompressionLevel = CompressionLevel.MODERATE, techniques: Optional[List[str]] = None) -> str` | 823 | Compress text using specified level and techniques. |
| 🔧 | `compress_to_tokens` | `(text: str, target_tokens: int, model: str = 'gpt-4') -> str` | 844 | Compress text to fit within token budget. |
| 🔧 | `estimate_compression` | `(text: str, level: CompressionLevel = CompressionLevel.MODERATE) -> Dict[str, Any]` | 866 | Estimate compression results without modifying text. |

---

## `argus/core/llm/anthropic.py`

### 🏛️ `class AnthropicLLM(BaseLLM)`  <sub>line 24</sub>

> Anthropic (Claude) LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'claude-3-5-sonnet-20241022', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, timeout: float = 60.0, max_retries: int = 3, **kwargs: Any)` | 64 | Initialize Anthropic LLM provider. |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 104 | Return provider name. |
| ⚙️ | `_init_client` | `(self) -> None` | 108 | Initialize the Anthropic client. |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 130 | Generate a response from Anthropic. |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 225 | Stream generated tokens from Anthropic. |
| ⚙️ | `_prepare_anthropic_messages` | `(self, prompt: str \| list[Message]) -> list[dict[str, Any]]` | 283 | Prepare messages in Anthropic format. |
| ⚙️ | `count_tokens` | `(self, text: str) -> int` | 320 | Count tokens using Anthropic's tokenizer. |

---

## `argus/core/llm/azure_openai.py`

### 🏛️ `class AzureOpenAILLM(BaseLLM)`  <sub>line 18</sub>

> Azure OpenAI LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'gpt-4', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, azure_endpoint: Optional[str] = None, api_version: str = '2024-02-15-preview', azure_deployment: Optional[str] = None, **kwargs: Any)` | 32 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 50 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 53 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 66 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 92 |  |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 108 |  |

---

## `argus/core/llm/base.py`

### 🏛️ `class MessageRole(str, Enum)`  <sub>line 22</sub>

> Role of a message in a conversation.  

### 🏛️ `class Message(BaseModel)`  <sub>line 31</sub>

> A single message in a conversation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@classmethod` | `system` | `(cls, content: str) -> 'Message'` | 67 | Create a system message. |
| ⚙️ `@classmethod` | `user` | `(cls, content: str) -> 'Message'` | 72 | Create a user message. |
| ⚙️ `@classmethod` | `assistant` | `(cls, content: str) -> 'Message'` | 77 | Create an assistant message. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 81 | Convert to provider-agnostic dictionary. |

### 🏛️ `class LLMConfig(BaseModel)`  <sub>line 93</sub>

> Configuration for LLM generation.  

### 🏛️ `class LLMUsage()`  <sub>line 173</sub>

> Decorators: `@dataclass`  
> Token usage statistics from LLM call.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__add__` | `(self, other: 'LLMUsage') -> 'LLMUsage'` | 179 | Add usage statistics. |

### 🏛️ `class LLMResponse()`  <sub>line 189</sub>

> Decorators: `@dataclass`  
> Response from an LLM generation call.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `success` | `(self) -> bool` | 211 | Check if response was successful. |

### 🏛️ `class BaseLLM(ABC)`  <sub>line 216</sub>

> Abstract base class for LLM providers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str, api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 230 | Initialize the LLM provider. |
| ⚙️ `@property`, `@abstractmethod` | `provider_name` | `(self) -> str` | 257 | Return the provider name. |
| ⚙️ `@abstractmethod` | `_init_client` | `(self) -> None` | 262 | Initialize the provider client. |
| ⚙️ `@abstractmethod` | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 267 | Generate a response from the LLM. |
| ⚙️ `@abstractmethod` | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 294 | Stream generated tokens from the LLM. |
| ⚙️ | `embed` | `(self, texts: str \| list[str], **kwargs: Any) -> list[list[float]]` | 320 | Generate embeddings for text(s). |
| ⚙️ | `count_tokens` | `(self, text: str) -> int` | 343 | Count tokens in text. |
| ⚙️ | `_prepare_messages` | `(self, prompt: str \| list[Message], system_prompt: Optional[str] = None) -> list[dict[str, Any]]` | 365 | Prepare messages for API call. |
| ⚙️ | `_measure_latency` | `(self, start_time: float) -> float` | 395 | Calculate latency in milliseconds. |
| ⚙️ | `__repr__` | `(self) -> str` | 399 | String representation. |

### 🏛️ `class EmbeddingModel()`  <sub>line 404</sub>

> Dedicated embedding model wrapper.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None, normalize: bool = True)` | 417 | Initialize embedding model. |
| ⚙️ | `_load_model` | `(self) -> None` | 437 | Lazy load the model. |
| ⚙️ `@property` | `dimension` | `(self) -> int` | 461 | Get embedding dimension. |
| ⚙️ | `embed` | `(self, texts: str \| list[str], batch_size: int = 32, show_progress: bool = False) -> list[list[float]]` | 466 | Generate embeddings for text(s). |
| ⚙️ | `embed_query` | `(self, query: str) -> list[float]` | 501 | Embed a query (single text). |
| ⚙️ | `embed_documents` | `(self, documents: list[str], batch_size: int = 32) -> list[list[float]]` | 513 | Embed multiple documents. |

---

## `argus/core/llm/bedrock.py`

### 🏛️ `class BedrockLLM(BaseLLM)`  <sub>line 19</sub>

> AWS Bedrock LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'anthropic.claude-3-5-sonnet-20241022-v2:0', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, region: str = 'us-east-1', profile_name: Optional[str] = None, **kwargs: Any)` | 42 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 59 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 62 |  |
| ⚙️ | `_get_model_family` | `(self) -> str` | 75 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 86 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 139 |  |

---

## `argus/core/llm/cerebras.py`

### 🏛️ `class CerebrasLLM(BaseLLM)`  <sub>line 18</sub>

> Cerebras LLM provider - fastest inference.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'llama3.1-70b', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 36 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 49 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 52 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 62 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 88 |  |

---

## `argus/core/llm/cloudflare.py`

### 🏛️ `class CloudflareLLM(BaseLLM)`  <sub>line 18</sub>

> Cloudflare Workers AI LLM provider - edge inference.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = '@cf/meta/llama-3.1-70b-instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, account_id: Optional[str] = None, **kwargs: Any)` | 35 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 50 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 53 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 65 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 95 |  |

---

## `argus/core/llm/cohere.py`

### 🏛️ `class CohereLLM(BaseLLM)`  <sub>line 24</sub>

> Cohere LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'command-r-plus', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 52 | Initialize Cohere LLM provider. |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 85 | Return provider name. |
| ⚙️ | `_init_client` | `(self) -> None` | 89 | Initialize the Cohere client. |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 107 | Generate a response from Cohere. |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 199 | Stream generated tokens from Cohere. |
| ⚙️ | `_prepare_cohere_messages` | `(self, prompt: str \| list[Message], system_prompt: Optional[str] = None) -> list[dict[str, Any]]` | 253 | Prepare messages in Cohere format. |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, input_type: str = 'search_document', **kwargs: Any) -> list[list[float]]` | 302 | Generate embeddings using Cohere. |
| ⚙️ | `count_tokens` | `(self, text: str) -> int` | 349 | Count tokens using Cohere's tokenizer. |

---

## `argus/core/llm/databricks.py`

### 🏛️ `class DatabricksLLM(BaseLLM)`  <sub>line 18</sub>

> Databricks LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'databricks-dbrx-instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, host: Optional[str] = None, **kwargs: Any)` | 29 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 43 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 46 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 58 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 84 |  |

---

## `argus/core/llm/deepseek.py`

### 🏛️ `class DeepSeekLLM(BaseLLM)`  <sub>line 24</sub>

> DeepSeek LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'deepseek-chat', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, base_url: Optional[str] = None, timeout: float = 60.0, **kwargs: Any)` | 50 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 67 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 70 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 81 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 129 |  |

---

## `argus/core/llm/fireworks.py`

### 🏛️ `class FireworksLLM(BaseLLM)`  <sub>line 18</sub>

> Fireworks AI LLM provider - fast inference.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'accounts/fireworks/models/llama-v3p1-70b-instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 38 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 51 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 54 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 64 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 90 |  |

---

## `argus/core/llm/gemini.py`

### 🏛️ `class GeminiLLM(BaseLLM)`  <sub>line 24</sub>

> Google Gemini LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'gemini-1.5-pro', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 53 | Initialize Gemini LLM provider. |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 86 | Return provider name. |
| ⚙️ | `_init_client` | `(self) -> None` | 90 | Initialize the Gemini client. |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 111 | Generate a response from Gemini. |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 199 | Stream generated tokens from Gemini. |
| ⚙️ | `_prepare_gemini_content` | `(self, prompt: str \| list[Message], system_prompt: Optional[str] = None) -> list[dict[str, Any]]` | 252 | Prepare content in Gemini format. |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 296 | Generate embeddings using Gemini. |
| ⚙️ | `count_tokens` | `(self, text: str) -> int` | 335 | Count tokens using Gemini's tokenizer. |

---

## `argus/core/llm/groq.py`

### 🏛️ `class GroqLLM(BaseLLM)`  <sub>line 24</sub>

> Groq LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'llama-3.1-70b-versatile', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 57 | Initialize Groq LLM provider. |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 90 | Return provider name. |
| ⚙️ | `_init_client` | `(self) -> None` | 94 | Initialize the Groq client. |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, response_format: Optional[dict[str, str]] = None, seed: Optional[int] = None, **kwargs: Any) -> LLMResponse` | 112 | Generate a response from Groq. |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 201 | Stream generated tokens from Groq. |
| ⚙️ | `transcribe` | `(self, audio_file: str \| bytes, model: str = 'whisper-large-v3', language: Optional[str] = None, **kwargs: Any) -> str` | 254 | Transcribe audio using Groq's Whisper. |
| ⚙️ | `list_models` | `(self) -> list[dict[str, Any]]` | 304 | List available models on Groq. |

---

## `argus/core/llm/huggingface.py`

### 🏛️ `class HuggingFaceLLM(BaseLLM)`  <sub>line 18</sub>

> HuggingFace Inference API LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'meta-llama/Llama-3.1-70B-Instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, endpoint_url: Optional[str] = None, **kwargs: Any)` | 29 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 43 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 46 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 56 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 84 |  |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 100 |  |

---

## `argus/core/llm/litellm.py`

### 🏛️ `class LiteLLMLLM(BaseLLM)`  <sub>line 18</sub>

> LiteLLM universal proxy provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'gpt-4', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, api_base: Optional[str] = None, **kwargs: Any)` | 30 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 44 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 47 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 64 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 95 |  |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 112 |  |

---

## `argus/core/llm/llamacpp.py`

### 🏛️ `class LlamaCppLLM(BaseLLM)`  <sub>line 18</sub>

> Llama.cpp local LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'llama', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, model_path: Optional[str] = None, n_ctx: int = 4096, n_gpu_layers: int = -1, n_threads: Optional[int] = None, **kwargs: Any)` | 29 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 49 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 52 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 74 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 108 |  |

---

## `argus/core/llm/mistral.py`

### 🏛️ `class MistralLLM(BaseLLM)`  <sub>line 24</sub>

> Mistral AI LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'mistral-large-latest', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 58 | Initialize Mistral LLM provider. |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 91 | Return provider name. |
| ⚙️ | `_init_client` | `(self) -> None` | 95 | Initialize the Mistral client. |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, safe_prompt: bool = False, **kwargs: Any) -> LLMResponse` | 113 | Generate a response from Mistral. |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 206 | Stream generated tokens from Mistral. |
| ⚙️ | `_prepare_mistral_messages` | `(self, prompt: str \| list[Message], system_prompt: Optional[str] = None) -> list[dict[str, Any]]` | 260 | Prepare messages in Mistral format. |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 303 | Generate embeddings using Mistral. |

---

## `argus/core/llm/nvidia.py`

### 🏛️ `class NvidiaLLM(BaseLLM)`  <sub>line 18</sub>

> NVIDIA AI Endpoints LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'meta/llama-3.1-70b-instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, base_url: Optional[str] = None, **kwargs: Any)` | 31 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 45 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 48 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 58 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 84 |  |

---

## `argus/core/llm/ollama.py`

### 🏛️ `class OllamaLLM(BaseLLM)`  <sub>line 27</sub>

> Ollama LLM provider for local models.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'llama3.2', host: str = 'http://localhost:11434', temperature: float = 0.7, max_tokens: int = 4096, timeout: float = 120.0, **kwargs: Any)` | 58 | Initialize Ollama LLM provider. |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 96 | Return provider name. |
| ⚙️ | `_init_client` | `(self) -> None` | 100 | Initialize the HTTP client. |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 122 | Generate a response from Ollama. |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 220 | Stream generated tokens from Ollama. |
| ⚙️ | `_prepare_ollama_messages` | `(self, prompt: str \| list[Message], system_prompt: Optional[str] = None) -> list[dict[str, str]]` | 292 | Prepare messages in Ollama format. |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 324 | Generate embeddings using Ollama. |
| ⚙️ | `list_models` | `(self) -> list[str]` | 367 | List available models on the Ollama server. |
| ⚙️ | `pull_model` | `(self, model_name: str) -> bool` | 383 | Pull a model from Ollama registry. |

---

## `argus/core/llm/openai.py`

### 🏛️ `class OpenAILLM(BaseLLM)`  <sub>line 24</sub>

> OpenAI LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'gpt-4', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, organization: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 60.0, max_retries: int = 3, **kwargs: Any)` | 60 | Initialize OpenAI LLM provider. |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 106 | Return provider name. |
| ⚙️ | `_init_client` | `(self) -> None` | 110 | Initialize the OpenAI client. |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, response_format: Optional[dict[str, str]] = None, seed: Optional[int] = None, **kwargs: Any) -> LLMResponse` | 138 | Generate a response from OpenAI. |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 235 | Stream generated tokens from OpenAI. |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 288 | Generate embeddings using OpenAI. |
| ⚙️ | `count_tokens` | `(self, text: str) -> int` | 327 | Count tokens using tiktoken for accurate OpenAI tokenization. |

---

## `argus/core/llm/perplexity.py`

### 🏛️ `class PerplexityLLM(BaseLLM)`  <sub>line 18</sub>

> Perplexity LLM provider with search grounding.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'llama-3.1-sonar-large-128k-online', api_key: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 4096, **kwargs: Any)` | 40 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 53 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 56 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, return_citations: bool = False, **kwargs: Any) -> LLMResponse` | 66 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 93 |  |

---

## `argus/core/llm/registry.py`

### 🏛️ `class LLMRegistry()`  <sub>line 19</sub>

> Registry for LLM providers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@classmethod` | `register` | `(cls, name: str, provider_class: Type[BaseLLM]) -> None` | 43 | Register an LLM provider. |
| ⚙️ `@classmethod` | `get` | `(cls, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None, cache: bool = True, **kwargs: Any) -> BaseLLM` | 55 | Get an LLM provider instance. |
| ⚙️ `@classmethod` | `list_providers` | `(cls) -> list[str]` | 137 | List registered provider names. |
| ⚙️ `@classmethod` | `has_provider` | `(cls, name: str) -> bool` | 147 | Check if a provider is registered. |
| ⚙️ `@classmethod` | `clear_cache` | `(cls) -> None` | 160 | Clear the instance cache. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `_register_default_providers` | `() -> None` | 166 | Register the built-in LLM providers. |
| 🔧 | `get_llm` | `(provider: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> BaseLLM` | 254 | Get an LLM provider instance. |
| 🔧 | `register_provider` | `(name: str, provider_class: Type[BaseLLM]) -> None` | 288 | Register a custom LLM provider. |
| 🔧 | `list_providers` | `() -> list[str]` | 308 | List available LLM providers. |

---

## `argus/core/llm/replicate.py`

### 🏛️ `class ReplicateLLM(BaseLLM)`  <sub>line 18</sub>

> Replicate LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'meta/meta-llama-3.1-405b-instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 35 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 48 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 51 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 62 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 90 |  |

---

## `argus/core/llm/sambanova.py`

### 🏛️ `class SambanovaLLM(BaseLLM)`  <sub>line 18</sub>

> SambaNova Cloud LLM provider - ultra-fast inference.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'Meta-Llama-3.1-70B-Instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 37 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 50 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 53 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 63 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 89 |  |

---

## `argus/core/llm/snowflake.py`

### 🏛️ `class SnowflakeLLM(BaseLLM)`  <sub>line 18</sub>

> Snowflake Cortex LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'llama3.1-70b', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, account: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, warehouse: Optional[str] = None, database: Optional[str] = None, schema: str = 'PUBLIC', **kwargs: Any)` | 35 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 60 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 63 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 79 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 117 |  |

---

## `argus/core/llm/together.py`

### 🏛️ `class TogetherLLM(BaseLLM)`  <sub>line 18</sub>

> Together AI LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'meta-llama/Llama-3.3-70B-Instruct-Turbo', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 38 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 51 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 54 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 64 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 90 |  |
| ⚙️ | `embed` | `(self, texts: str \| list[str], model: Optional[str] = None, **kwargs: Any) -> list[list[float]]` | 106 |  |

---

## `argus/core/llm/vertex.py`

### 🏛️ `class VertexAILLM(BaseLLM)`  <sub>line 18</sub>

> Google Vertex AI LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'gemini-1.5-pro', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 8192, project: Optional[str] = None, location: str = 'us-central1', **kwargs: Any)` | 34 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 51 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 54 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 66 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 102 |  |

---

## `argus/core/llm/vllm.py`

### 🏛️ `class VllmLLM(BaseLLM)`  <sub>line 18</sub>

> vLLM self-hosted LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'meta-llama/Llama-3.1-70B-Instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, base_url: str = 'http://localhost:8000/v1', **kwargs: Any)` | 29 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 43 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 46 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 56 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 82 |  |

---

## `argus/core/llm/watsonx.py`

### 🏛️ `class WatsonxLLM(BaseLLM)`  <sub>line 18</sub>

> IBM watsonx.ai LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'ibm/granite-3-8b-instruct', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, project_id: Optional[str] = None, url: str = 'https://us-south.ml.cloud.ibm.com', **kwargs: Any)` | 34 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 51 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 54 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 68 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 104 |  |

---

## `argus/core/llm/xai.py`

### 🏛️ `class XaiLLM(BaseLLM)`  <sub>line 18</sub>

> xAI (Grok) LLM provider.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model: str = 'grok-2', api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 4096, **kwargs: Any)` | 31 |  |
| ⚙️ `@property` | `provider_name` | `(self) -> str` | 43 |  |
| ⚙️ | `_init_client` | `(self) -> None` | 46 |  |
| ⚙️ | `generate` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> LLMResponse` | 56 |  |
| ⚙️ | `stream` | `(self, prompt: str \| list[Message], *, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[list[str]] = None, **kwargs: Any) -> Iterator[str]` | 83 |  |

---

## `argus/core/models.py`

### 🏛️ `class SourceType(str, Enum)`  <sub>line 47</sub>

> Type of document source.  

### 🏛️ `class Document(BaseModel)`  <sub>line 58</sub>

> Represents an ingested document with metadata.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `content_hash` | `(self) -> str` | 134 | Compute content hash if not provided. |
| ⚙️ `@computed_field`, `@property` | `word_count` | `(self) -> int` | 140 | Count words in document content. |
| ⚙️ | `__hash__` | `(self) -> int` | 144 | Hash by document ID for set operations. |
| ⚙️ | `__eq__` | `(self, other: object) -> bool` | 148 | Equality based on document ID. |

### 🏛️ `class Chunk(BaseModel)`  <sub>line 155</sub>

> Represents a segment of a document for embedding and retrieval.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `span` | `(self) -> tuple[int, int]` | 224 | Character span tuple. |
| ⚙️ `@computed_field`, `@property` | `length` | `(self) -> int` | 230 | Character length of chunk. |
| ⚙️ | `__hash__` | `(self) -> int` | 234 | Hash by chunk ID. |
| ⚙️ | `__eq__` | `(self, other: object) -> bool` | 238 | Equality based on chunk ID. |

### 🏛️ `class Embedding(BaseModel)`  <sub>line 245</sub>

> Represents a vector embedding for a chunk or query.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `dimension` | `(self) -> int` | 288 | Vector dimension. |
| ⚙️ | `__hash__` | `(self) -> int` | 292 | Hash by embedding ID. |

### 🏛️ `class CitationType(str, Enum)`  <sub>line 297</sub>

> Type of citation.  

### 🏛️ `class Citation(BaseModel)`  <sub>line 305</sub>

> Represents a citation linking a claim to its source.  

### 🏛️ `class Claim(BaseModel)`  <sub>line 354</sub>

> Represents an extracted claim with citations and confidence.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@field_validator('confidence')`, `@classmethod` | `validate_confidence` | `(cls, v: float) -> float` | 414 | Ensure confidence is properly bounded. |
| ⚙️ | `__hash__` | `(self) -> int` | 418 | Hash by claim ID. |

### 🏛️ `class NodeType(str, Enum)`  <sub>line 423</sub>

> Types of nodes in the C-DAG.  

### 🏛️ `class NodeBase(BaseModel)`  <sub>line 432</sub>

> Base class for all C-DAG nodes.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__hash__` | `(self) -> int` | 488 | Hash by node ID. |
| ⚙️ | `__eq__` | `(self, other: object) -> bool` | 492 | Equality based on node ID. |
| ⚙️ | `to_prov_dict` | `(self) -> dict[str, Any]` | 498 | Convert node to PROV-O compatible dictionary. |

### 🏛️ `class ScoredItem(BaseModel)`  <sub>line 515</sub>

> Generic scored item for retrieval results.  

### 🏛️ `class RetrievalResult(BaseModel)`  <sub>line 553</sub>

> Result from hybrid retrieval query.  

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_uuid` | `() -> str` | 26 | Generate a unique identifier. |
| 🔧 | `compute_hash` | `(content: str, algorithm: str = 'sha256') -> str` | 31 | Compute cryptographic hash of content. |

---

## `argus/core/openapi.py`

### 🏛️ `class AuthType(Enum)`  <sub>line 22</sub>

> Authentication types supported by OpenAPI.  

### 🏛️ `class SecurityScheme()`  <sub>line 32</sub>

> Decorators: `@dataclass`  
> Security scheme configuration.  

### 🏛️ `class APIParameter()`  <sub>line 43</sub>

> Decorators: `@dataclass`  
> API parameter definition.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `param_type` | `(self) -> str` | 53 |  |

### 🏛️ `class APIOperation()`  <sub>line 58</sub>

> Decorators: `@dataclass`  
> API operation definition.  

### 🏛️ `class APISpec()`  <sub>line 74</sub>

> Decorators: `@dataclass`  
> Parsed OpenAPI specification.  

### 🏛️ `class OpenAPIParser()`  <sub>line 86</sub>

> Parser for OpenAPI/Swagger specifications.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 89 |  |
| ⚙️ | `parse` | `(self, spec: Dict[str, Any]) -> APISpec` | 92 | Parse OpenAPI specification. |
| ⚙️ | `_parse_security_scheme` | `(self, name: str, scheme_def: Dict[str, Any]) -> SecurityScheme` | 159 | Parse a security scheme definition. |
| ⚙️ | `_parse_parameters` | `(self, params: List[Dict[str, Any]]) -> List[APIParameter]` | 187 | Parse parameter definitions. |
| ⚙️ | `_parse_operation` | `(self, path: str, method: str, op_def: Dict[str, Any], path_params: List[APIParameter], is_v3: bool) -> APIOperation` | 207 | Parse an operation definition. |
| ⚙️ | `_resolve_ref` | `(self, ref: str) -> Dict[str, Any]` | 268 | Resolve a $ref reference. |

### 🏛️ `class OpenAPIClient()`  <sub>line 285</sub>

> Dynamic REST API client generated from OpenAPI specification.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, spec: APISpec, base_url: Optional[str] = None, auth_config: Optional[Dict[str, Any]] = None, timeout: int = 30)` | 297 |  |
| ⚙️ `@classmethod` | `from_dict` | `(cls, spec_dict: Dict[str, Any], base_url: Optional[str] = None, auth_config: Optional[Dict[str, Any]] = None, timeout: int = 30) -> 'OpenAPIClient'` | 315 | Create client from specification dictionary. |
| ⚙️ `@classmethod` | `from_json` | `(cls, json_str: str, base_url: Optional[str] = None, auth_config: Optional[Dict[str, Any]] = None, timeout: int = 30) -> 'OpenAPIClient'` | 328 | Create client from JSON string. |
| ⚙️ `@classmethod` | `from_file` | `(cls, file_path: str, base_url: Optional[str] = None, auth_config: Optional[Dict[str, Any]] = None, timeout: int = 30) -> 'OpenAPIClient'` | 340 | Create client from specification file. |
| ⚙️ `@classmethod` | `from_url` | `(cls, url: str, base_url: Optional[str] = None, auth_config: Optional[Dict[str, Any]] = None, timeout: int = 30) -> 'OpenAPIClient'` | 363 | Create client from specification URL. |
| ⚙️ | `_get_session` | `(self)` | 389 | Get HTTP session with authentication configured. |
| ⚙️ | `_configure_auth` | `(self)` | 400 | Configure session authentication. |
| ⚙️ | `_setup_methods` | `(self)` | 432 | Setup dynamic methods for each operation. |
| ⚙️ | `_to_snake_case` | `(self, name: str) -> str` | 444 | Convert camelCase/PascalCase to snake_case. |
| ⚙️ | `_create_method` | `(self, operation: APIOperation) -> Callable` | 450 | Create a method for an operation. |
| ⚙️ | `method` | `(**kwargs)` | 452 |  |
| ⚙️ | `execute` | `(self, operation_id: str, **kwargs) -> Dict[str, Any]` | 474 | Execute an API operation. |
| ⚙️ | `_execute_operation` | `(self, operation: APIOperation, **kwargs) -> Dict[str, Any]` | 492 | Execute an operation. |
| ⚙️ | `list_operations` | `(self) -> List[Dict[str, Any]]` | 571 | List all available operations. |
| ⚙️ | `get_operation_schema` | `(self, operation_id: str) -> Dict[str, Any]` | 587 | Get detailed schema for an operation. |

### 🏛️ `class OpenAPIToolGenerator()`  <sub>line 620</sub>

> Generate ARGUS tools from OpenAPI specifications.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, spec: APISpec)` | 627 |  |
| ⚙️ `@classmethod` | `from_url` | `(cls, url: str) -> 'OpenAPIToolGenerator'` | 632 | Create generator from specification URL. |
| ⚙️ `@classmethod` | `from_file` | `(cls, file_path: str) -> 'OpenAPIToolGenerator'` | 645 | Create generator from specification file. |
| ⚙️ | `generate_tool_class` | `(self, class_name: Optional[str] = None, operations: Optional[List[str]] = None) -> str` | 661 | Generate Python source code for a BaseTool subclass. |
| ⚙️ | `_generate_action_method` | `(self, operation: APIOperation) -> List[str]` | 800 | Generate an action method for an operation. |
| ⚙️ | `_to_snake_case` | `(self, name: str) -> str` | 879 | Convert to snake_case. |
| ⚙️ | `_to_env_var` | `(self, name: str) -> str` | 884 | Convert to environment variable format. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `load_openapi_spec` | `(source: str) -> APISpec` | 891 | Load an OpenAPI specification from various sources. |
| 🔧 | `create_client` | `(source: str, base_url: Optional[str] = None, auth_config: Optional[Dict[str, Any]] = None) -> OpenAPIClient` | 922 | Create an OpenAPI client from various sources. |
| 🔧 | `generate_tool_code` | `(source: str, class_name: Optional[str] = None, output_file: Optional[str] = None) -> str` | 942 | Generate tool code from OpenAPI specification. |

---

## `argus/crux/agent_card.py`

### 🏛️ `class AgentCalibration(BaseModel)`  <sub>line 35</sub>

> Calibration metrics for an agent.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `is_well_calibrated` | `(self) -> bool` | 84 | Check if agent is well-calibrated (ECE < 0.10). |
| ⚙️ `@computed_field`, `@property` | `is_credible` | `(self) -> bool` | 90 | Check if agent has sufficient credibility (> 0.50). |
| ⚙️ `@computed_field`, `@property` | `effective_weight` | `(self) -> float` | 96 | Compute effective weight for BRP merges. |
| ⚙️ `@classmethod` | `neutral` | `(cls) -> 'AgentCalibration'` | 106 | Create neutral calibration for new agents. |
| ⚙️ `@classmethod` | `from_metrics` | `(cls, metrics: 'CalibrationMetrics', current_credibility: float = 0.5) -> 'AgentCalibration'` | 117 | Create from ARGUS CalibrationMetrics. |
| ⚙️ | `update` | `(self, new_brier: float, recency_weight: float = 0.15) -> 'AgentCalibration'` | 140 | Update calibration with new observation. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 177 | Convert to dictionary. |

### 🏛️ `class AgentCapabilities(BaseModel)`  <sub>line 191</sub>

> Capabilities declared by an agent.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@classmethod` | `specialist` | `(cls, domains: list[str]) -> 'AgentCapabilities'` | 233 | Create capabilities for a Specialist agent. |
| ⚙️ `@classmethod` | `refuter` | `(cls, domains: list[str]) -> 'AgentCapabilities'` | 244 | Create capabilities for a Refuter agent. |
| ⚙️ `@classmethod` | `jury` | `(cls) -> 'AgentCapabilities'` | 255 | Create capabilities for a Jury agent. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 265 | Convert to dictionary. |

### 🏛️ `class EpistemicAgentCard(BaseModel)`  <sub>line 276</sub>

> Epistemic Agent Card (EAC) - CRUX Primitive 1.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `is_expired` | `(self) -> bool` | 370 | Check if card has expired. |
| ⚙️ `@computed_field`, `@property` | `card_hash` | `(self) -> str` | 378 | Compute hash of card for verification. |
| ⚙️ `@classmethod` | `from_argus_agent` | `(cls, agent: 'BaseAgent', belief_domains: Optional[list[str]] = None, crux_endpoint: Optional[str] = None) -> 'EpistemicAgentCard'` | 390 | Create EAC from existing ARGUS agent. |
| ⚙️ | `can_handle_domain` | `(self, domain: str) -> bool` | 438 | Check if agent can handle a specific domain. |
| ⚙️ | `domain_match_score` | `(self, domains: list[str]) -> float` | 461 | Compute domain match score against a list of domains. |
| ⚙️ | `refresh_calibration` | `(self, new_calibration: AgentCalibration) -> 'EpistemicAgentCard'` | 484 | Create new card with updated calibration. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 512 | Convert to dictionary (compatible with A2A Agent Card format). |
| ⚙️ | `to_json` | `(self, indent: int = 2) -> str` | 532 | Convert to JSON string. |
| ⚙️ `@classmethod` | `from_json` | `(cls, json_str: str) -> 'EpistemicAgentCard'` | 537 | Parse from JSON string. |

### 🏛️ `class EACRegistry()`  <sub>line 576</sub>

> Registry for Epistemic Agent Cards.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 584 | Initialize empty registry. |
| ⚙️ | `register` | `(self, card: EpistemicAgentCard) -> None` | 590 | Register an agent card. |
| ⚙️ | `unregister` | `(self, agent_id: str) -> bool` | 612 | Unregister an agent card. |
| ⚙️ | `get` | `(self, agent_id: str) -> Optional[EpistemicAgentCard]` | 639 | Get card by agent ID. |
| ⚙️ | `get_card` | `(self, agent_id: str) -> Optional[EpistemicAgentCard]` | 643 | Get card by agent ID (alias for get). |
| ⚙️ | `get_by_type` | `(self, agent_type: str) -> list[EpistemicAgentCard]` | 647 | Get all cards of a specific type. |
| ⚙️ | `get_by_domain` | `(self, domain: str) -> list[EpistemicAgentCard]` | 652 | Get all cards handling a specific domain. |
| ⚙️ | `get_challengers` | `(self, exclude_agent: str, domain: Optional[str] = None) -> list[EpistemicAgentCard]` | 657 | Get agents capable of challenging claims. |
| ⚙️ | `get_all` | `(self) -> list[EpistemicAgentCard]` | 688 | Get all registered cards. |
| ⚙️ | `__len__` | `(self) -> int` | 692 | Number of registered cards. |
| ⚙️ | `__contains__` | `(self, agent_id: str) -> bool` | 696 | Check if agent is registered. |

---

## `argus/crux/auction.py`

### 🏛️ `class BidRejectionReason(str, Enum)`  <sub>line 41</sub>

> Reasons for rejecting a bid.  

### 🏛️ `class BidResult()`  <sub>line 52</sub>

> Decorators: `@dataclass`  
> Result of submitting a bid.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 65 | Convert to dictionary. |

### 🏛️ `class AuctionResult()`  <sub>line 75</sub>

> Decorators: `@dataclass`  
> Result of a Challenger Auction.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 102 |  |
| ⚙️ `@property` | `num_bids` | `(self) -> int` | 107 | Number of bids received. |
| ⚙️ `@property` | `duration_seconds` | `(self) -> float` | 112 | Auction duration in seconds. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 118 | Convert to dictionary. |

### 🏛️ `class BidEvaluator()`  <sub>line 135</sub>

> Evaluates bids in a Challenger Auction.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, router: Optional['DialecticalRouter'] = None, config: Optional[CRUXConfig] = None)` | 152 | Initialize evaluator. |
| ⚙️ | `validate_bid` | `(self, bid: ChallengerBid, claim_bundle: 'ClaimBundle', credibility_floor: Optional[float] = None, existing_bidders: Optional[set[str]] = None) -> tuple[bool, Optional[BidRejectionReason]]` | 167 | Validate a bid. |
| ⚙️ | `compute_bid_score` | `(self, bid: ChallengerBid, claim_bundle: 'ClaimBundle', agent_card: Optional['EpistemicAgentCard'] = None) -> float` | 205 | Compute score for a bid. |
| ⚙️ | `rank_bids` | `(self, bids: list[ChallengerBid], claim_bundle: 'ClaimBundle', agent_cards: Optional[dict[str, 'EpistemicAgentCard']] = None) -> list[tuple[ChallengerBid, float]]` | 249 | Rank bids by score. |
| ⚙️ | `select_winner` | `(self, ranked_bids: list[tuple[ChallengerBid, float]]) -> Optional[ChallengerBid]` | 281 | Select winning bid from ranked list. |

### 🏛️ `class ChallengerAuction()`  <sub>line 301</sub>

> Challenger Auction - CRUX Primitive 7.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, router: Optional['DialecticalRouter'] = None, registry: Optional['EACRegistry'] = None, config: Optional[CRUXConfig] = None)` | 322 | Initialize auction. |
| ⚙️ | `start` | `(self, claim_bundle: 'ClaimBundle', timeout_seconds: Optional[int] = None) -> 'AuctionSession'` | 351 | Start a Challenger Auction for a Claim Bundle. |
| ⚙️ | `get_session` | `(self, auction_id: str) -> Optional['AuctionSession']` | 390 | Get auction session by ID. |
| ⚙️ | `get_session_for_cb` | `(self, cb_id: str) -> Optional['AuctionSession']` | 394 | Get active auction session for a Claim Bundle. |
| ⚙️ | `submit_bid` | `(self, auction_id: str, bid: ChallengerBid) -> BidResult` | 401 | Submit a bid to an auction. |
| ⚙️ | `close` | `(self, auction_id: str, force: bool = False) -> AuctionResult` | 434 | Close an auction and determine winner. |
| ⚙️ | `close_if_timeout` | `(self, auction_id: str) -> Optional[AuctionResult]` | 477 | Close auction if timeout reached. |
| ⚙️ | `⚡ async run_auction` | `(self, claim_bundle: 'ClaimBundle', timeout_seconds: Optional[int] = None, bid_source: Optional[Callable] = None) -> AuctionResult` | 496 | Run a complete auction asynchronously. |
| ⚙️ | `get_result` | `(self, auction_id: str) -> Optional[AuctionResult]` | 533 | Get result for a completed auction. |
| ⚙️ | `on_bid_received` | `(self, callback: Callable) -> None` | 537 | Register callback for bid received events. |
| ⚙️ | `on_auction_complete` | `(self, callback: Callable) -> None` | 541 | Register callback for auction complete events. |
| ⚙️ `@property` | `active_auctions` | `(self) -> list[str]` | 546 | List of active auction IDs. |

### 🏛️ `class AuctionSession()`  <sub>line 551</sub>

> An active auction session.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, auction_id: str, claim_bundle: 'ClaimBundle', timeout_seconds: int, evaluator: BidEvaluator, registry: Optional['EACRegistry'] = None)` | 558 | Initialize auction session. |
| ⚙️ `@property` | `cb_id` | `(self) -> str` | 591 | Claim Bundle ID. |
| ⚙️ `@property` | `state` | `(self) -> AuctionState` | 596 | Current auction state. |
| ⚙️ `@property` | `is_expired` | `(self) -> bool` | 601 | Check if auction has expired. |
| ⚙️ `@property` | `time_remaining` | `(self) -> float` | 606 | Time remaining in seconds. |
| ⚙️ `@property` | `num_bids` | `(self) -> int` | 612 | Number of bids received. |
| ⚙️ | `submit_bid` | `(self, bid: ChallengerBid) -> BidResult` | 616 | Submit a bid to this auction. |
| ⚙️ | `close` | `(self, force: bool = False) -> AuctionResult` | 667 | Close auction and determine winner. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `create_bid` | `(agent_id: str, cb_id: str, estimated_dfs: float, domain_confidence: float = 0.5, strategy: str = 'general_critique', expires_in_seconds: int = 300) -> ChallengerBid` | 731 | Helper function to create a ChallengerBid. |

---

## `argus/crux/brp.py`

### 🏛️ `class ContradictionType(str, Enum)`  <sub>line 43</sub>

> Types of contradictions between Claim Bundles.  

### 🏛️ `class Contradiction()`  <sub>line 51</sub>

> Decorators: `@dataclass`  
> Detected contradiction between two Claim Bundles.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `proposition_id` | `(self) -> str` | 69 | Get the proposition ID (same for both bundles). |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 73 | Convert to dictionary. |

### 🏛️ `class MiniDebateRound()`  <sub>line 85</sub>

> Decorators: `@dataclass`  
> A round in the BRP mini-debate.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 102 | Convert to dictionary. |

### 🏛️ `class BRPSession()`  <sub>line 116</sub>

> Decorators: `@dataclass`  
> A Belief Reconciliation Protocol session.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `is_active` | `(self) -> bool` | 147 | Check if session is active. |
| ⚙️ `@property` | `current_round` | `(self) -> int` | 152 | Get current round number. |
| ⚙️ | `transition` | `(self, new_state: BRPState) -> None` | 156 | Transition to a new state. |
| ⚙️ | `add_round` | `(self, round_data: MiniDebateRound) -> None` | 171 | Add a mini-debate round. |
| ⚙️ `@property` | `duration_seconds` | `(self) -> float` | 176 | Get session duration in seconds. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 183 | Convert to dictionary. |

### 🏛️ `class ContradictionDetector()`  <sub>line 210</sub>

> Detects contradictions between Claim Bundles.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, threshold: float = 0.2)` | 218 | Initialize detector. |
| ⚙️ | `detect` | `(self, bundle_a: ClaimBundle, bundle_b: ClaimBundle) -> Optional[Contradiction]` | 227 | Detect contradiction between two bundles. |
| ⚙️ | `find_contradictions` | `(self, bundles: list[ClaimBundle]) -> list[Contradiction]` | 277 | Find all contradictions in a list of bundles. |

### 🏛️ `class BayesianMerger()`  <sub>line 311</sub>

> Performs Bayesian merge of contradicting Claim Bundles.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@staticmethod` | `compute_merged_posterior` | `(posteriors: list[float], credibilities: list[float]) -> float` | 321 | Compute credibility-weighted merged posterior. |
| ⚙️ `@staticmethod` | `merge_distributions` | `(distributions: list[ConfidenceDistribution], weights: list[float]) -> ConfidenceDistribution` | 349 | Merge distributions via moment matching. |

### 🏛️ `class ReconciliationResult()`  <sub>line 393</sub>

> Decorators: `@dataclass`  
> Result of Belief Reconciliation Protocol.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 408 | Convert to dictionary. |

### 🏛️ `class BeliefReconciliationProtocol()`  <sub>line 423</sub>

> Belief Reconciliation Protocol (BRP) - CRUX Primitive 4.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[CRUXConfig] = None, registry: Optional['EACRegistry'] = None, ledger: Optional['CredibilityLedger'] = None)` | 440 | Initialize BRP. |
| ⚙️ | `detect_contradiction` | `(self, bundle_a: ClaimBundle, bundle_b: ClaimBundle) -> Optional[Contradiction]` | 466 | Detect contradiction between two bundles. |
| ⚙️ | `find_all_contradictions` | `(self, bundles: list[ClaimBundle]) -> list[Contradiction]` | 483 | Find all contradictions in a list of bundles. |
| ⚙️ | `start_session` | `(self, contradiction: Contradiction) -> BRPSession` | 490 | Start a new BRP session. |
| ⚙️ | `run_mini_debate` | `(self, session: BRPSession, round_claims: Optional[dict[str, str]] = None, round_evidence: Optional[dict[str, list[str]]] = None, round_distributions: Optional[dict[str, ConfidenceDistribution]] = None) -> MiniDebateRound` | 522 | Run a mini-debate round. |
| ⚙️ | `perform_merge` | `(self, session: BRPSession) -> ClaimBundle` | 554 | Perform Bayesian merge of contradicting bundles. |
| ⚙️ | `complete_session` | `(self, session: BRPSession, merged_bundle: ClaimBundle) -> BRPResolution` | 606 | Complete a BRP session with provenance fork. |
| ⚙️ | `reconcile` | `(self, bundles_or_contradiction: Union[Contradiction, list[ClaimBundle]], max_rounds: Optional[int] = None) -> ReconciliationResult` | 671 | Full reconciliation workflow for a contradiction. |
| ⚙️ | `get_session` | `(self, session_id: str) -> Optional[BRPSession]` | 749 | Get active session by ID. |
| ⚙️ | `get_history` | `(self, proposition_id: Optional[str] = None) -> list[BRPSession]` | 753 | Get history of completed reconciliations. |

---

## `argus/crux/claim_bundle.py`

### 🏛️ `class ClaimBundle(BaseModel)`  <sub>line 42</sub>

> Claim Bundle (CB) - CRUX Primitive 2.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, **data)` | 193 | Initialize and compute CB ID if not provided. |
| ⚙️ | `_compute_cb_id` | `(self) -> str` | 214 | Compute CB ID from content hash. |
| ⚙️ | `_compute_lineage_hash` | `(self) -> str` | 227 | Compute argument lineage hash. |
| ⚙️ `@computed_field`, `@property` | `is_expired` | `(self) -> bool` | 238 | Check if bundle has expired. |
| ⚙️ `@computed_field`, `@property` | `challenge_expired` | `(self) -> bool` | 245 | Check if challenge window has closed. |
| ⚙️ `@computed_field`, `@property` | `is_supporting` | `(self) -> bool` | 253 | Check if bundle supports proposition. |
| ⚙️ `@computed_field`, `@property` | `is_attacking` | `(self) -> bool` | 259 | Check if bundle attacks proposition. |
| ⚙️ `@computed_field`, `@property` | `effective_weight` | `(self) -> float` | 265 | Compute effective weight for BRP merges. |
| ⚙️ `@computed_field`, `@property` | `is_merged` | `(self) -> bool` | 277 | Check if this is a merged/reconciled bundle. |
| ⚙️ | `close_challenge` | `(self) -> None` | 281 | Close the bundle for challenges. |
| ⚙️ | `mark_unchallenged` | `(self) -> None` | 285 | Mark as unchallenged (no challengers responded). |
| ⚙️ | `conflicts_with` | `(self, other: 'ClaimBundle', threshold: float = 0.2) -> bool` | 291 | Check if this bundle conflicts with another. |
| ⚙️ | `verify_signature` | `(self, public_key: str) -> bool` | 321 | Verify Ed25519 signature. |
| ⚙️ | `_get_signable_content` | `(self) -> str` | 353 | Get content for signing/verification. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 365 | Convert to dictionary (CRUX wire format). |
| ⚙️ | `to_json` | `(self, indent: int = 2) -> str` | 391 | Convert to JSON string. |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'ClaimBundle'` | 396 | Create from dictionary. |

### 🏛️ `class ClaimBundleFactory()`  <sub>line 437</sub>

> Factory for creating Claim Bundles from ARGUS entities.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@staticmethod` | `from_evidence` | `(evidence: 'Evidence', proposition_id: str, agent_card: 'EpistemicAgentCard', prior: float = 0.5) -> ClaimBundle` | 446 | Create Claim Bundle from ARGUS Evidence node. |
| ⚙️ `@staticmethod` | `from_proposition_posterior` | `(proposition: 'Proposition', agent_card: 'EpistemicAgentCard', evidence_refs: Optional[list[str]] = None) -> ClaimBundle` | 504 | Create Claim Bundle from Proposition posterior. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `merge_claim_bundles` | `(bundles: list[ClaimBundle], credibility_weights: Optional[list[float]] = None) -> ClaimBundle` | 551 | Merge multiple Claim Bundles into a reconciled bundle. |

---

## `argus/crux/edr.py`

### 🏛️ `class EDRCheckpoint()`  <sub>line 43</sub>

> Decorators: `@dataclass`  
> A checkpoint in the EDR system.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 68 |  |
| ⚙️ | `_compute_hash` | `(self) -> str` | 72 | Compute checkpoint hash. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 82 | Convert to dictionary. |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'EDRCheckpoint'` | 95 | Create from dictionary. |

### 🏛️ `class BeliefConflict()`  <sub>line 110</sub>

> Decorators: `@dataclass`  
> A conflict between agent's checkpoint beliefs and current state.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 127 |  |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 130 | Convert to dictionary. |

### 🏛️ `class SyncResult()`  <sub>line 142</sub>

> Decorators: `@dataclass`  
> Result of EDR synchronization.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 161 | Convert to dictionary. |

### 🏛️ `class EDRSynchronizer()`  <sub>line 173</sub>

> Synchronizer for Epistemic Dead Reckoning.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, ledger: Optional['CredibilityLedger'] = None, brp: Optional['BeliefReconciliationProtocol'] = None, config: Optional[CRUXConfig] = None)` | 190 | Initialize synchronizer. |
| ⚙️ | `create_checkpoint` | `(self, agent_id: str, claim_bundles: Optional[list['ClaimBundle']] = None) -> EDRCheckpoint` | 217 | Create a checkpoint for an agent. |
| ⚙️ | `find_checkpoint` | `(self, checkpoint_hash: str) -> Optional[EDRCheckpoint]` | 261 | Find a checkpoint by hash. |
| ⚙️ | `get_latest_checkpoint` | `(self, agent_id: str) -> Optional[EDRCheckpoint]` | 280 | Get most recent checkpoint for an agent. |
| ⚙️ | `register_bundle` | `(self, bundle: 'ClaimBundle') -> None` | 290 | Register an active Claim Bundle. |
| ⚙️ | `compute_delta` | `(self, agent_id: str, last_checkpoint_hash: str) -> EDRDelta` | 298 | Compute delta from a checkpoint to current state. |
| ⚙️ | `detect_conflicts` | `(self, checkpoint: EDRCheckpoint, threshold: float = 0.2) -> list[BeliefConflict]` | 367 | Detect conflicts between checkpoint beliefs and current state. |
| ⚙️ | `sync` | `(self, agent_id: str, last_checkpoint_hash: str) -> SyncResult` | 408 | Full synchronization for a reconnecting agent. |

### 🏛️ `class EpistemicDeadReckoning()`  <sub>line 492</sub>

> Epistemic Dead Reckoning (EDR) - CRUX Primitive 6.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, ledger: Optional['CredibilityLedger'] = None, brp: Optional['BeliefReconciliationProtocol'] = None, config: Optional[CRUXConfig] = None)` | 512 | Initialize EDR. |
| ⚙️ `@property` | `enabled` | `(self) -> bool` | 531 | Check if EDR is enabled. |
| ⚙️ | `checkpoint` | `(self, agent_id: str, claim_bundles: Optional[list['ClaimBundle']] = None) -> EDRCheckpoint` | 535 | Create a checkpoint for an agent. |
| ⚙️ | `reconnect` | `(self, agent_id: str, last_checkpoint_hash: str) -> SyncResult` | 555 | Reconnect an agent using EDR. |
| ⚙️ | `register_bundle` | `(self, bundle: 'ClaimBundle') -> None` | 575 | Register a new Claim Bundle. |
| ⚙️ | `get_checkpoint` | `(self, agent_id: str) -> Optional[EDRCheckpoint]` | 579 | Get latest checkpoint for an agent. |

---

## `argus/crux/ledger.py`

### 🏛️ `class CredibilityEntry()`  <sub>line 39</sub>

> Decorators: `@dataclass`  
> A single entry in the Credibility Ledger.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `compute_hash` | `(self) -> str` | 70 | Compute SHA-256 hash for this entry. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 85 | Convert to dictionary. |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'CredibilityEntry'` | 101 | Create from dictionary. |

### 🏛️ `class CredibilityUpdate()`  <sub>line 119</sub>

> Decorators: `@dataclass`  
> Update to an agent's credibility.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `delta` | `(self) -> float` | 139 | Change in credibility. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 143 | Convert to dictionary. |

### 🏛️ `class AgentCredibilityState()`  <sub>line 157</sub>

> Decorators: `@dataclass`  
> Current credibility state for an agent.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `average_brier` | `(self) -> float` | 179 | Average Brier score. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 185 | Convert to dictionary. |

### 🏛️ `class CredibilityLedger()`  <sub>line 201</sub>

> Credibility Ledger (CL) - CRUX Primitive 5.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[CRUXConfig] = None, path: Optional[str] = None, backend: str = 'memory')` | 232 | Initialize Credibility Ledger. |
| ⚙️ `@property` | `head_hash` | `(self) -> str` | 268 | Get current head hash of the chain. |
| ⚙️ `@property` | `chain_length` | `(self) -> int` | 275 | Get number of entries in chain. |
| ⚙️ | `_get_or_create_state` | `(self, agent_id: str) -> AgentCredibilityState` | 279 | Get or create agent state. |
| ⚙️ | `record_prediction` | `(self, agent_id: str, proposition_id: str, predicted_posterior: float, session_id: str = '', metadata: Optional[dict[str, Any]] = None) -> CredibilityEntry` | 285 | Record a prediction (awaiting outcome). |
| ⚙️ | `record_outcome` | `(self, entry_id: str, ground_truth: int) -> Optional[CredibilityUpdate]` | 327 | Record outcome for a pending prediction. |
| ⚙️ | `record_prediction_with_outcome` | `(self, agent_id: str, proposition_id: str, predicted_posterior: float, ground_truth: int, session_id: str = '') -> CredibilityUpdate` | 378 | Record prediction and outcome together. |
| ⚙️ | `_update_credibility` | `(self, agent_id: str, brier_contribution: float) -> CredibilityUpdate` | 428 | Update agent credibility using EMA. |
| ⚙️ | `_check_suspension` | `(self, state: AgentCredibilityState) -> bool` | 468 | Check if agent should be suspended. |
| ⚙️ | `get_credibility` | `(self, agent_id: str) -> Optional[float]` | 488 | Get current credibility for an agent. |
| ⚙️ | `get_agent_state` | `(self, agent_id: str) -> Optional[AgentCredibilityState]` | 503 | Get full state for an agent. |
| ⚙️ | `is_suspended` | `(self, agent_id: str) -> bool` | 507 | Check if agent is suspended. |
| ⚙️ | `reinstate_agent` | `(self, agent_id: str) -> bool` | 512 | Reinstate a suspended agent. |
| ⚙️ | `get_entries_for_agent` | `(self, agent_id: str, limit: Optional[int] = None) -> list[CredibilityEntry]` | 531 | Get ledger entries for an agent. |
| ⚙️ | `get_entries_since` | `(self, timestamp: datetime) -> list[CredibilityEntry]` | 554 | Get all entries since a timestamp. |
| ⚙️ | `get_entries_after_hash` | `(self, hash_value: str) -> list[CredibilityEntry]` | 561 | Get all entries after a specific hash. |
| ⚙️ | `verify_chain_integrity` | `(self) -> tuple[bool, Optional[str]]` | 577 | Verify hash chain integrity. |
| ⚙️ | `detect_sybil_agents` | `(self, min_samples: int = 10, suspicion_threshold: float = 0.95) -> list[str]` | 600 | Detect potential Sybil agents. |
| ⚙️ | `_append_to_file` | `(self, entry: CredibilityEntry) -> None` | 636 | Append entry to ledger file. |
| ⚙️ | `_load` | `(self) -> None` | 641 | Load ledger from file. |
| ⚙️ | `get_all_agent_ids` | `(self) -> list[str]` | 672 | Get all agent IDs with credibility records. |
| ⚙️ | `get_credibility_history` | `(self, agent_id: str) -> list[dict[str, Any]]` | 676 | Get credibility history for an agent. |
| ⚙️ | `export_statistics` | `(self) -> dict[str, Any]` | 711 | Export ledger statistics. |

---

## `argus/crux/models.py`

### 🏛️ `class CRUXVersion(str, Enum)`  <sub>line 38</sub>

> CRUX protocol versions.  

### 🏛️ `class Polarity(int, Enum)`  <sub>line 43</sub>

> Claim polarity relative to proposition.  

### 🏛️ `class DistributionType(str, Enum)`  <sub>line 50</sub>

> Types of confidence distributions.  

### 🏛️ `class BRPState(str, Enum)`  <sub>line 57</sub>

> States of the Belief Reconciliation Protocol.  

### 🏛️ `class AuctionState(str, Enum)`  <sub>line 67</sub>

> States of the Challenger Auction.  

### 🏛️ `class BetaDistribution(BaseModel)`  <sub>line 76</sub>

> Beta distribution for uncertainty representation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@computed_field`, `@property` | `mean` | `(self) -> float` | 107 | Compute distribution mean. |
| ⚙️ `@computed_field`, `@property` | `mode` | `(self) -> float` | 113 | Compute distribution mode (for alpha, beta > 1). |
| ⚙️ `@computed_field`, `@property` | `variance` | `(self) -> float` | 121 | Compute distribution variance. |
| ⚙️ `@computed_field`, `@property` | `std` | `(self) -> float` | 128 | Compute distribution standard deviation. |
| ⚙️ `@computed_field`, `@property` | `concentration` | `(self) -> float` | 134 | Total concentration (alpha + beta). |
| ⚙️ | `pdf` | `(self, x: float) -> float` | 138 | Probability density function at x. |
| ⚙️ | `cdf` | `(self, x: float) -> float` | 161 | Cumulative distribution function at x. |
| ⚙️ | `sample` | `(self, n: int = 1) -> list[float]` | 174 | Draw samples from the distribution. |
| ⚙️ | `quantile` | `(self, q: float) -> float` | 187 | Compute quantile (inverse CDF). |
| ⚙️ | `credible_interval` | `(self, level: float = 0.95) -> tuple[float, float]` | 200 | Compute credible interval. |
| ⚙️ `@classmethod` | `from_mean_concentration` | `(cls, mean: float, concentration: float) -> 'BetaDistribution'` | 214 | Create from mean and concentration. |
| ⚙️ `@classmethod` | `from_mean_std` | `(cls, mean: float, std: float) -> 'BetaDistribution'` | 234 | Create from mean and standard deviation. |
| ⚙️ `@classmethod` | `uniform` | `(cls) -> 'BetaDistribution'` | 261 | Create uniform (non-informative) distribution. |
| ⚙️ `@classmethod` | `jeffreys` | `(cls) -> 'BetaDistribution'` | 266 | Create Jeffreys prior (minimally informative). |
| ⚙️ | `update` | `(self, successes: int = 0, failures: int = 0) -> 'BetaDistribution'` | 270 | Bayesian update with new observations. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 290 | Convert to dictionary. |

### 🏛️ `class ConfidenceDistribution(BaseModel)`  <sub>line 301</sub>

> Wrapper for confidence distributions of various types.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@classmethod` | `from_beta` | `(cls, alpha: float, beta: float) -> 'ConfidenceDistribution'` | 339 | Create from Beta parameters. |
| ⚙️ `@classmethod` | `from_point` | `(cls, value: float) -> 'ConfidenceDistribution'` | 354 | Create point estimate (no uncertainty). |
| ⚙️ `@classmethod` | `from_mean_std` | `(cls, mean: float, std: float) -> 'ConfidenceDistribution'` | 363 | Create from mean and standard deviation. |
| ⚙️ | `sample` | `(self, n: int = 1) -> list[float]` | 377 | Draw samples from distribution. |
| ⚙️ | `credible_interval` | `(self, level: float = 0.95) -> tuple[float, float]` | 388 | Compute credible interval. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 401 | Convert to dictionary. |

### 🏛️ `class ChallengerBid()`  <sub>line 415</sub>

> Decorators: `@dataclass`  
> Bid submitted by an agent to challenge a Claim Bundle.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 438 |  |
| ⚙️ `@property` | `is_expired` | `(self) -> bool` | 443 | Check if bid has expired. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 447 | Convert to dictionary. |

### 🏛️ `class BRPResolution()`  <sub>line 462</sub>

> Decorators: `@dataclass`  
> Result of Belief Reconciliation Protocol.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 489 | Convert to dictionary. |

### 🏛️ `class EDRDelta()`  <sub>line 509</sub>

> Decorators: `@dataclass`  
> Delta bundle for Epistemic Dead Reckoning synchronization.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `num_changes` | `(self) -> int` | 533 | Total number of changes in delta. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 541 | Convert to dictionary. |

### 🏛️ `class CRUXConfig(BaseModel)`  <sub>line 554</sub>

> Configuration for CRUX protocol.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `validate_dfs_weights` | `(self) -> bool` | 654 | Validate that DFS weights sum to approximately 1.0. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 664 | Convert to dictionary. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_crux_id` | `(prefix: str) -> str` | 26 | Generate a unique CRUX identifier with prefix. |
| 🔧 | `compute_crux_hash` | `(content: str, algorithm: str = 'sha256') -> str` | 31 | Compute cryptographic hash for CRUX entities. |

---

## `argus/crux/orchestrator.py`

### 🏛️ `class CRUXSessionState(str, Enum)`  <sub>line 80</sub>

> States of a CRUX session.  

### 🏛️ `class CRUXSessionStats()`  <sub>line 92</sub>

> Decorators: `@dataclass`  
> Statistics for a CRUX session.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 113 | Convert to dictionary. |

### 🏛️ `class CRUXSession()`  <sub>line 127</sub>

> Decorators: `@dataclass`  
> A CRUX debate session.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `duration_seconds` | `(self) -> float` | 167 | Session duration in seconds. |
| ⚙️ | `add_claim_bundle` | `(self, cb: ClaimBundle) -> None` | 172 | Add a Claim Bundle to the session. |
| ⚙️ | `add_auction` | `(self, result: AuctionResult) -> None` | 182 | Add an auction result to the session. |
| ⚙️ | `add_brp_session` | `(self, brp: BRPSession) -> None` | 192 | Add a BRP session to the session. |
| ⚙️ | `get_claim_bundles_for_proposition` | `(self, prop_id: str) -> list[ClaimBundle]` | 197 | Get all Claim Bundles for a proposition. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 202 | Convert to dictionary. |

### 🏛️ `class CRUXDebateResult()`  <sub>line 220</sub>

> Decorators: `@dataclass`  
> Result of a CRUX-enabled debate.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `was_challenged` | `(self) -> bool` | 248 | Check if the final claim was challenged. |
| ⚙️ `@property` | `credibility_impact` | `(self) -> dict[str, float]` | 253 | Get credibility changes for each agent. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 260 | Convert to dictionary. |

### 🏛️ `class CRUXOrchestrator()`  <sub>line 281</sub>

> CRUX-enabled orchestrator for ARGUS debates.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, base: Optional['RDCOrchestrator'] = None, llm: Optional['BaseLLM'] = None, ledger: Optional[CredibilityLedger] = None, config: Optional[CRUXConfig] = None, contradiction_threshold: float = 0.2, auction_timeout_seconds: int = 30, enable_edr: bool = True)` | 307 | Initialize CRUX orchestrator. |
| ⚙️ | `_setup_agent_cards` | `(self) -> None` | 379 | Set up Epistemic Agent Cards for base agents. |
| ⚙️ | `debate` | `(self, proposition_text: str, prior: float = 0.5, retriever: Optional['HybridRetriever'] = None, domain: str = 'general') -> CRUXDebateResult` | 441 | Run a CRUX-enabled debate on a proposition. |
| ⚙️ | `_update_agent_domains` | `(self, proposition_text: str, domain: str) -> None` | 584 | Update agent domain beliefs based on proposition. |
| ⚙️ | `_convert_to_claim_bundles` | `(self, result: 'DebateResult', proposition_text: str) -> list[ClaimBundle]` | 604 | Convert debate evidence to Claim Bundles. |
| ⚙️ | `_compute_evidence_posterior` | `(self, prior: float, likelihood_ratio: float, confidence: float) -> float` | 705 | Compute posterior from evidence. |
| ⚙️ | `_run_auction` | `(self, claim_bundle: ClaimBundle) -> Optional[AuctionResult]` | 736 | Run a Challenger Auction for a Claim Bundle. |
| ⚙️ | `_get_potential_challengers` | `(self, claim_bundle: ClaimBundle) -> list[tuple[str, float]]` | 778 | Get potential challengers for a Claim Bundle. |
| ⚙️ | `_reconcile_claims` | `(self, claim_bundles: list[ClaimBundle], proposition_id: str) -> Optional[ClaimBundle]` | 811 | Reconcile potentially contradicting Claim Bundles. |
| ⚙️ | `_find_contradictions` | `(self, claim_bundles: list[ClaimBundle]) -> list[tuple[str, str]]` | 875 | Find contradicting Claim Bundle pairs. |
| ⚙️ | `_update_credibility` | `(self, claim_bundles: list[ClaimBundle], verdict: 'Verdict') -> None` | 908 | Update Credibility Ledger based on debate outcome. |
| ⚙️ | `on_claim_bundle` | `(self, callback: Callable[[ClaimBundle], None]) -> None` | 947 | Register callback for Claim Bundle events. |
| ⚙️ | `on_brp_triggered` | `(self, callback: Callable[[BRPSession], None]) -> None` | 951 | Register callback for BRP trigger events. |
| ⚙️ | `on_auction_complete` | `(self, callback: Callable[[AuctionResult], None]) -> None` | 955 | Register callback for auction complete events. |
| ⚙️ `@property` | `current_session` | `(self) -> Optional[CRUXSession]` | 961 | Get current CRUX session. |
| ⚙️ `@property` | `session_history` | `(self) -> list[CRUXSession]` | 966 | Get all completed sessions. |
| ⚙️ | `register_agent` | `(self, card: EpistemicAgentCard) -> None` | 971 | Register an Epistemic Agent Card. |
| ⚙️ | `get_agent_card` | `(self, agent_id: str) -> Optional[EpistemicAgentCard]` | 976 | Get agent card by ID. |
| ⚙️ | `get_card_endpoint` | `(self, agent_id: str) -> dict[str, Any]` | 981 | GET /crux/card - Return Epistemic Agent Card. |
| ⚙️ | `submit_claim_endpoint` | `(self, claim_data: dict[str, Any]) -> dict[str, Any]` | 988 | POST /crux/claim - Submit a Claim Bundle. |
| ⚙️ | `submit_bid_endpoint` | `(self, bid_data: dict[str, Any]) -> dict[str, Any]` | 1002 | POST /crux/bid - Submit a Challenger Bid. |
| ⚙️ | `get_ledger_endpoint` | `(self, agent_id: str) -> dict[str, Any]` | 1024 | GET /crux/ledger/{agent} - Get Credibility Ledger for agent. |
| ⚙️ | `sync_endpoint` | `(self, sync_data: dict[str, Any]) -> dict[str, Any]` | 1036 | POST /crux/sync - EDR resync request. |

---

## `argus/crux/routing.py`

### 🏛️ `class DialecticalFitnessScore()`  <sub>line 35</sub>

> Decorators: `@dataclass`  
> Dialectical Fitness Score (DFS) for an agent-claim pair.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 61 | Convert to dictionary. |

### 🏛️ `class DomainMatcher()`  <sub>line 77</sub>

> Matches agent belief domains against claim text.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@classmethod` | `extract_domains_from_text` | `(cls, text: str) -> list[str]` | 106 | Extract domain indicators from claim text. |
| ⚙️ `@classmethod` | `compute_domain_match` | `(cls, agent_domains: list[str], claim_domains: list[str]) -> float` | 126 | Compute domain match score. |

### 🏛️ `class AdversarialAssessor()`  <sub>line 177</sub>

> Assesses adversarial potential of an agent against a claim.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@staticmethod` | `compute_adversarial_potential` | `(agent_prior: Optional[float], claim_posterior: float, claim_polarity: Polarity) -> float` | 186 | Compute adversarial potential score. |

### 🏛️ `class RecencyTracker()`  <sub>line 227</sub>

> Tracks recent agent engagements to avoid over-engagement.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, decay_hours: float = 1.0)` | 235 | Initialize tracker. |
| ⚙️ | `record_engagement` | `(self, agent_id: str, proposition_id: str) -> None` | 245 | Record an agent engagement. |
| ⚙️ | `compute_recency_penalty` | `(self, agent_id: str, proposition_id: str) -> float` | 259 | Compute recency penalty for agent-proposition pair. |
| ⚙️ | `cleanup_old` | `(self, max_age_hours: float = 24.0) -> int` | 289 | Remove old engagement records. |

### 🏛️ `class DialecticalRouter()`  <sub>line 377</sub>

> Routes Claim Bundles to agents using Dialectical Fitness Scores.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, registry: 'EACRegistry', config: Optional[CRUXConfig] = None)` | 390 | Initialize router. |
| ⚙️ | `set_agent_prior` | `(self, agent_id: str, proposition_id: str, prior: float) -> None` | 409 | Record an agent's prior belief on a proposition. |
| ⚙️ | `get_agent_prior` | `(self, agent_id: str, proposition_id: str) -> Optional[float]` | 425 | Get agent's prior belief on a proposition. |
| ⚙️ | `rank_challengers` | `(self, claim_bundle: 'ClaimBundle', top_k: Optional[int] = None) -> list[DialecticalFitnessScore]` | 438 | Rank potential challengers by DFS. |
| ⚙️ | `get_best_challenger` | `(self, claim_bundle: 'ClaimBundle') -> Optional[DialecticalFitnessScore]` | 491 | Get the best challenger for a Claim Bundle. |
| ⚙️ | `record_challenge` | `(self, agent_id: str, proposition_id: str) -> None` | 507 | Record that an agent has challenged a proposition. |
| ⚙️ | `get_routing_table` | `(self, claim_bundle: 'ClaimBundle') -> dict[str, DialecticalFitnessScore]` | 519 | Get full routing table for a Claim Bundle. |
| ⚙️ | `visualize_routing` | `(self, claim_bundle: 'ClaimBundle') -> dict[str, Any]` | 535 | Get visualization data for routing decisions. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `compute_dfs` | `(agent: 'EpistemicAgentCard', claim_bundle: 'ClaimBundle', config: Optional[CRUXConfig] = None, agent_prior: Optional[float] = None, recency_penalty: float = 0.0) -> DialecticalFitnessScore` | 315 | Compute Dialectical Fitness Score for an agent-claim pair. |

---

## `argus/crux/visualization.py`

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `_check_plotly` | `() -> None` | 97 | Check if Plotly is available. |
| 🔧 | `_apply_dark_layout` | `(fig: Any, title: str = '') -> Any` | 106 | Apply consistent dark styling to a figure. |
| 🔧 | `_get_polarity_color` | `(polarity: int) -> str` | 115 | Get color based on claim polarity. |
| 🔧 | `_get_credibility_color` | `(credibility: float) -> str` | 124 | Get color based on credibility rating. |
| 🔧 | `plot_crux_debate_flow` | `(session: 'CRUXSession', show_auctions: bool = True, show_brp: bool = True, interactive: bool = True) -> Any` | 137 | Plot the entire CRUX debate as an interactive flow DAG. |
| 🔧 | `plot_credibility_evolution` | `(ledger: 'CredibilityLedger', agent_ids: Optional[list[str]] = None, num_points: int = 50) -> Any` | 419 | Plot credibility rating evolution over time for agents. |
| 🔧 | `plot_brp_merge` | `(brp_session: 'BRPSession', show_distributions: bool = True) -> Any` | 524 | Visualize a Belief Reconciliation Protocol merge. |
| 🔧 | `plot_dfs_heatmap` | `(dfs_scores: dict[str, dict[str, float]], title: str = 'Dialectical Fitness Scores') -> Any` | 655 | Create heatmap of Dialectical Fitness Scores. |
| 🔧 | `plot_dfs_breakdown` | `(dfs_result: Any) -> Any` | 734 | Plot breakdown of a single DFS score components. |
| 🔧 | `plot_auction_results` | `(auction_result: 'AuctionResult') -> Any` | 792 | Visualize Challenger Auction results. |
| 🔧 | `plot_auction_timeline` | `(auctions: list['AuctionResult']) -> Any` | 894 | Plot timeline of multiple auctions in a session. |
| 🔧 | `create_crux_dashboard` | `(session: 'CRUXSession', ledger: Optional['CredibilityLedger'] = None) -> Any` | 971 | Create comprehensive CRUX dashboard with multiple visualizations. |
| 🔧 | `export_debate_static` | `(result: 'CRUXDebateResult', output_path: str, format: str = 'png', width: int = 1200, height: int = 800) -> str` | 1084 | Export debate visualization as static image. |
| 🔧 | `export_session_json` | `(session: 'CRUXSession', output_path: str) -> str` | 1139 | Export session data as JSON for external visualization. |
| 🔧 | `build_dfs_matrix` | `(session: 'CRUXSession', router: Any) -> dict[str, dict[str, float]]` | 1176 | Build DFS matrix from session and router. |

---

## `argus/debate/visualization.py`

### 🏛️ `class ArgumentType(Enum)`  <sub>line 99</sub>

> Types of arguments in a debate.  

### 🏛️ `class ArgumentStatus(Enum)`  <sub>line 109</sub>

> Status of an argument.  

### 🏛️ `class Argument()`  <sub>line 119</sub>

> Decorators: `@dataclass`  
> Represents an argument in a debate.  

### 🏛️ `class DebateRound()`  <sub>line 134</sub>

> Decorators: `@dataclass`  
> Represents a round of debate.  

### 🏛️ `class DebateSession()`  <sub>line 146</sub>

> Decorators: `@dataclass`  
> Represents a complete debate session.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `all_arguments` | `(self) -> List[Argument]` | 158 | Get all arguments from all rounds. |
| ⚙️ `@property` | `duration_seconds` | `(self) -> float` | 166 | Get session duration in seconds. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `_check_plotly` | `() -> None` | 172 | Check if Plotly is available. |
| 🔧 | `_check_networkx` | `() -> None` | 181 | Check if NetworkX is available. |
| 🔧 | `plot_argument_flow` | `(session: DebateSession, layout: str = 'hierarchical', show_labels: bool = True, height: int = 700, width: int = 1000) -> 'go.Figure'` | 194 | Plot argument flow diagram showing relationships between arguments. |
| 🔧 | `_hierarchical_layout` | `(G: 'nx.DiGraph') -> Dict[str, Tuple[float, float]]` | 387 | Compute hierarchical layout for directed graph. |
| 🔧 | `plot_debate_timeline` | `(session: DebateSession, height: int = 500, width: int = 1000) -> 'go.Figure'` | 430 | Plot debate timeline showing arguments over time. |
| 🔧 | `plot_agent_performance` | `(session: DebateSession, height: int = 500, width: int = 800) -> 'go.Figure'` | 543 | Plot agent performance metrics. |
| 🔧 | `plot_confidence_evolution` | `(session: DebateSession, height: int = 400, width: int = 800) -> 'go.Figure'` | 707 | Plot how confidence scores evolve over the debate. |
| 🔧 | `plot_round_summary` | `(session: DebateSession, height: int = 400, width: int = 800) -> 'go.Figure'` | 813 | Plot summary of each debate round. |
| 🔧 | `plot_interaction_heatmap` | `(session: DebateSession, height: int = 500, width: int = 600) -> 'go.Figure'` | 938 | Plot heatmap of agent interactions. |
| 🔧 | `plot_argument_type_distribution` | `(session: DebateSession, height: int = 400, width: int = 400) -> 'go.Figure'` | 1005 | Plot distribution of argument types as pie chart. |
| 🔧 | `create_debate_dashboard` | `(session: DebateSession, height: int = 1200, width: int = 1400) -> 'go.Figure'` | 1065 | Create comprehensive debate dashboard with multiple visualizations. |
| 🔧 | `export_debate_html` | `(session: DebateSession, output_path: str, include_dashboard: bool = True) -> str` | 1267 | Export debate visualization as interactive HTML. |
| 🔧 | `export_debate_png` | `(session: DebateSession, output_path: str, width: int = 1400, height: int = 1000) -> str` | 1295 | Export debate visualization as PNG image. |
| 🔧 | `generate_debate_report` | `(session: DebateSession) -> Dict[str, Any]` | 1321 | Generate a JSON report of debate statistics. |

---

## `argus/decision/bayesian.py`

### 🏛️ `class UpdateResult()`  <sub>line 101</sub>

> Decorators: `@dataclass`  
> Result of a Bayesian update.  

### 🏛️ `class BayesianUpdater()`  <sub>line 113</sub>

> Bayesian updater for proposition beliefs.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, temperature: float = 1.0, regularization: float = 0.0)` | 129 | Initialize Bayesian updater. |
| ⚙️ | `update` | `(self, prior: float, evidence_contributions: list[float], weights: Optional[list[float]] = None) -> float` | 144 | Update belief given evidence contributions. |
| ⚙️ | `update_from_graph` | `(self, graph: 'CDAG', prop_id: str) -> UpdateResult` | 190 | Update proposition belief from C-DAG evidence. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `log_odds` | `(p: float) -> float` | 27 | Convert probability to log-odds. |
| 🔧 | `from_log_odds` | `(lo: float) -> float` | 44 | Convert log-odds to probability. |
| 🔧 | `compute_likelihood_ratio` | `(evidence_confidence: float, sensitivity: float = 0.8, specificity: float = 0.8) -> float` | 61 | Compute likelihood ratio for evidence. |
| 🔧 | `update_posterior` | `(prior: float, support_scores: list[tuple[float, float]], attack_scores: list[tuple[float, float]], temperature: float = 1.0) -> float` | 269 | Convenience function to update posterior given support/attack scores. |
| 🔧 | `batch_update` | `(priors: list[float], contributions: list[list[float]], temperature: float = 1.0) -> list[float]` | 311 | Batch update multiple propositions. |
| 🔧 | `sensitivity_analysis` | `(prior: float, base_contributions: list[float], target_index: int, delta_range: tuple[float, float] = (-1.0, 1.0), num_points: int = 20) -> list[tuple[float, float]]` | 337 | Analyze posterior sensitivity to a single evidence contribution. |

---

## `argus/decision/calibration.py`

### 🏛️ `class CalibrationMetrics()`  <sub>line 25</sub>

> Decorators: `@dataclass`  
> Calibration metrics for a set of predictions.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `summary` | `(self) -> str` | 44 | Get summary string. |

### 🏛️ `class ReliabilityDiagram()`  <sub>line 262</sub>

> Reliability diagram for visualizing calibration.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, num_bins: int = 10)` | 270 | Initialize reliability diagram. |
| ⚙️ | `add_sample` | `(self, confidence: float, outcome: int) -> None` | 284 | Add a prediction-outcome pair. |
| ⚙️ | `add_batch` | `(self, confidences: list[float], outcomes: list[int]) -> None` | 299 | Add multiple prediction-outcome pairs. |
| ⚙️ | `compute` | `(self) -> CalibrationMetrics` | 314 | Compute calibration metrics. |
| ⚙️ | `get_diagram_data` | `(self) -> dict[str, Any]` | 339 | Get data for plotting reliability diagram. |
| ⚙️ | `reset` | `(self) -> None` | 364 | Clear all samples. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `compute_brier_score` | `(confidences: list[float], outcomes: list[int]) -> float` | 54 | Compute Brier score. |
| 🔧 | `compute_ece` | `(confidences: list[float], outcomes: list[int], num_bins: int = 10) -> tuple[float, dict[str, Any]]` | 85 | Compute Expected Calibration Error. |
| 🔧 | `compute_nll` | `(confidences: list[float], outcomes: list[int], epsilon: float = 1e-10) -> float` | 151 | Compute Negative Log Likelihood. |
| 🔧 | `temperature_scaling` | `(logits: list[float], labels: list[int], initial_temp: float = 1.0) -> tuple[float, list[float]]` | 186 | Find optimal temperature for calibration. |
| 🔧 | `apply_temperature` | `(confidence: float, temperature: float) -> float` | 236 | Apply temperature scaling to a single confidence. |
| 🔧 | `calibrate_model_outputs` | `(train_confidences: list[float], train_outcomes: list[int], test_confidences: list[float]) -> tuple[float, list[float], CalibrationMetrics]` | 370 | Calibrate model outputs using temperature scaling. |

---

## `argus/decision/eig.py`

### 🏛️ `class ActionCandidate()`  <sub>line 29</sub>

> Decorators: `@dataclass`  
> A candidate action/experiment for EIG evaluation.  

### 🏛️ `class EIGEstimator()`  <sub>line 93</sub>

> Monte Carlo EIG estimator.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, num_samples: int = 1000, num_theta_samples: int = 100, seed: Optional[int] = None)` | 110 | Initialize EIG estimator. |
| ⚙️ | `estimate` | `(self, prior: float, outcome_model: Callable[[float], Any], update_model: Callable[[float, Any], float]) -> float` | 128 | Estimate EIG given outcome and update models. |
| ⚙️ | `estimate_discrete` | `(self, prior: float, outcomes: list[Any], outcome_probs: list[float], update_probs: list[float]) -> float` | 173 | Estimate EIG for discrete outcomes. |
| ⚙️ | `estimate_for_action` | `(self, graph: 'CDAG', action: ActionCandidate, prior_getter: Callable[[str], float], update_simulator: Callable[[str, dict], float]) -> float` | 210 | Estimate EIG for a specific action on a C-DAG. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `entropy` | `(p: float) -> float` | 56 | Compute binary entropy. |
| 🔧 | `kl_divergence` | `(p: float, q: float) -> float` | 73 | Compute KL divergence for binary distributions. |
| 🔧 | `estimate_eig` | `(prior: float, sensitivity: float = 0.8, specificity: float = 0.8) -> float` | 269 | Quick EIG estimate for a binary test. |
| 🔧 | `rank_actions_by_eig` | `(actions: list[ActionCandidate], priors: dict[str, float], sensitivity: float = 0.8, specificity: float = 0.8, cost_weight: float = 0.1) -> list[ActionCandidate]` | 314 | Rank actions by EIG-adjusted utility. |

---

## `argus/decision/planner.py`

### 🏛️ `class PlannerConfig()`  <sub>line 31</sub>

> Decorators: `@dataclass`  
> Configuration for VoI planner.  

### 🏛️ `class QueuedExperiment()`  <sub>line 52</sub>

> Decorators: `@dataclass`  
> An experiment in the priority queue.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__lt__` | `(self, other: 'QueuedExperiment') -> bool` | 58 |  |

### 🏛️ `class ExperimentQueue()`  <sub>line 63</sub>

> Priority queue for experiments.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_size: int = 10, budget: Optional[float] = None)` | 71 | Initialize experiment queue. |
| ⚙️ | `push` | `(self, action: ActionCandidate, priority: Optional[float] = None) -> bool` | 89 | Add an experiment to the queue. |
| ⚙️ | `pop` | `(self) -> Optional[ActionCandidate]` | 134 | Get and remove highest priority experiment. |
| ⚙️ | `peek` | `(self) -> Optional[ActionCandidate]` | 148 | Get highest priority without removing. |
| ⚙️ | `remove` | `(self, action_id: str) -> bool` | 159 | Remove an action from the queue. |
| ⚙️ | `get_all` | `(self) -> list[ActionCandidate]` | 179 | Get all actions in priority order. |
| ⚙️ | `__len__` | `(self) -> int` | 184 |  |
| ⚙️ `@property` | `remaining_budget` | `(self) -> Optional[float]` | 188 | Remaining budget. |

### 🏛️ `class VoIPlanner()`  <sub>line 195</sub>

> Value of Information based experiment planner.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[PlannerConfig] = None, eig_estimator: Optional[EIGEstimator] = None)` | 209 | Initialize planner. |
| ⚙️ | `plan` | `(self, graph: 'CDAG', candidates: list[ActionCandidate]) -> ExperimentQueue` | 224 | Create prioritized experiment queue. |
| ⚙️ | `_apply_risk_aversion` | `(self, actions: list[ActionCandidate], priors: dict[str, float]) -> list[ActionCandidate]` | 289 | Apply risk aversion to action utilities. |
| ⚙️ | `_knapsack_selection` | `(self, actions: list[ActionCandidate], budget: float) -> list[ActionCandidate]` | 320 | Select actions using 0/1 knapsack algorithm. |
| ⚙️ | `what_if_analysis` | `(self, graph: 'CDAG', action: ActionCandidate, outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]` | 376 | Analyze potential outcomes of an action. |
| ⚙️ | `stopping_criteria_met` | `(self, graph: 'CDAG', min_posterior: float = 0.95, max_eig: float = 0.01) -> tuple[bool, str]` | 433 | Check if stopping criteria are met. |

---

## `argus/durable/checkpointer.py`

### 🏛️ `class Checkpoint()`  <sub>line 23</sub>

> Decorators: `@dataclass`  
> A workflow checkpoint.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 33 |  |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'Checkpoint'` | 39 |  |

### 🏛️ `class BaseCheckpointer(ABC)`  <sub>line 45</sub>

> Abstract base class for checkpointers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@abstractmethod` | `save` | `(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str` | 49 |  |
| ⚙️ `@abstractmethod` | `load` | `(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]` | 53 |  |
| ⚙️ `@abstractmethod` | `list_checkpoints` | `(self, thread_id: str, limit: int = 10) -> List[Checkpoint]` | 57 |  |
| ⚙️ `@abstractmethod` | `delete` | `(self, checkpoint_id: str) -> bool` | 61 |  |

### 🏛️ `class MemoryCheckpointer(BaseCheckpointer)`  <sub>line 65</sub>

> In-memory checkpointer for development/testing.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_checkpoints: int = 100)` | 68 |  |
| ⚙️ | `save` | `(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str` | 73 |  |
| ⚙️ | `load` | `(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]` | 91 |  |
| ⚙️ | `list_checkpoints` | `(self, thread_id: str, limit: int = 10) -> List[Checkpoint]` | 99 |  |
| ⚙️ | `delete` | `(self, checkpoint_id: str) -> bool` | 103 |  |

### 🏛️ `class SQLiteCheckpointer(BaseCheckpointer)`  <sub>line 112</sub>

> SQLite-based persistent checkpointer.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, db_path: str = 'checkpoints.db')` | 115 |  |
| ⚙️ | `_init_db` | `(self) -> None` | 120 |  |
| ⚙️ | `save` | `(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str` | 135 |  |
| ⚙️ | `load` | `(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]` | 151 |  |
| ⚙️ | `list_checkpoints` | `(self, thread_id: str, limit: int = 10) -> List[Checkpoint]` | 168 |  |
| ⚙️ | `delete` | `(self, checkpoint_id: str) -> bool` | 179 |  |

### 🏛️ `class FileSystemCheckpointer(BaseCheckpointer)`  <sub>line 185</sub>

> File-based JSON checkpointer.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, base_path: str = './checkpoints')` | 188 |  |
| ⚙️ | `_get_thread_dir` | `(self, thread_id: str) -> Path` | 192 |  |
| ⚙️ | `save` | `(self, thread_id: str, state: dict[str, Any], step: int = 0, **metadata: Any) -> str` | 195 |  |
| ⚙️ | `load` | `(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]` | 208 |  |
| ⚙️ | `list_checkpoints` | `(self, thread_id: str, limit: int = 10) -> List[Checkpoint]` | 223 |  |
| ⚙️ | `delete` | `(self, checkpoint_id: str) -> bool` | 230 |  |

---

## `argus/durable/config.py`

### 🏛️ `class CheckpointerType(str, Enum)`  <sub>line 18</sub>

> Checkpointer storage backend types.  

### 🏛️ `class RetryPolicy(BaseModel)`  <sub>line 25</sub>

> Retry policy for failed tasks.  

### 🏛️ `class DurableConfig(BaseModel)`  <sub>line 33</sub>

> Configuration for durable execution.  

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_default_durable_config` | `() -> DurableConfig` | 46 |  |

---

## `argus/durable/state.py`

### 🏛️ `class StateSnapshot()`  <sub>line 22</sub>

> Decorators: `@dataclass`  
> A versioned state snapshot.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 30 |  |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'StateSnapshot'` | 36 |  |

### 🏛️ `class DebateState()`  <sub>line 43</sub>

> Decorators: `@dataclass`  
> Complete debate state for checkpointing.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 58 |  |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'DebateState'` | 65 |  |

### 🏛️ `class StateManager()`  <sub>line 111</sub>

> Manage debate state for durable execution.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 114 |  |
| ⚙️ | `initialize` | `(self, debate_id: str, proposition: str, max_rounds: int = 5) -> DebateState` | 119 | Initialize a new debate state. |
| ⚙️ | `get_state` | `(self) -> Optional[DebateState]` | 126 |  |
| ⚙️ | `update` | `(self, **updates: Any) -> DebateState` | 129 | Update current state. |
| ⚙️ | `snapshot` | `(self, description: str = '') -> StateSnapshot` | 139 | Create a snapshot of current state. |
| ⚙️ | `restore` | `(self, snapshot: StateSnapshot) -> DebateState` | 152 | Restore state from snapshot. |
| ⚙️ | `get_snapshots` | `(self) -> List[StateSnapshot]` | 157 |  |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `serialize_state` | `(state: DebateState) -> str` | 72 | Serialize debate state to JSON string. |
| 🔧 | `deserialize_state` | `(json_str: str) -> DebateState` | 77 | Deserialize debate state from JSON string. |
| 🔧 | `serialize_graph` | `(graph: 'CDAG') -> dict[str, Any]` | 82 | Serialize CDAG graph to dictionary. |

---

## `argus/durable/tasks.py`

### 🏛️ `class TaskResult()`  <sub>line 21</sub>

> Decorators: `@dataclass`  
> Result from a task execution.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 30 |  |

### 🏛️ `class TaskRegistry()`  <sub>line 41</sub>

> Registry for tracking executed tasks to prevent re-execution.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 44 |  |
| ⚙️ | `_compute_task_id` | `(self, name: str, args: tuple, kwargs: dict) -> str` | 48 | Compute deterministic task ID from name and arguments. |
| ⚙️ | `is_executed` | `(self, task_id: str) -> bool` | 53 |  |
| ⚙️ | `get_result` | `(self, task_id: str) -> Optional[TaskResult]` | 56 |  |
| ⚙️ | `record` | `(self, task_id: str, result: TaskResult) -> None` | 59 |  |
| ⚙️ | `mark_pending` | `(self, task_id: str) -> None` | 63 |  |
| ⚙️ | `is_pending` | `(self, task_id: str) -> bool` | 66 |  |
| ⚙️ | `clear` | `(self) -> None` | 69 |  |
| ⚙️ | `get_all_results` | `(self) -> Dict[str, TaskResult]` | 73 |  |

### 🏛️ `class TaskExecutor()`  <sub>line 141</sub>

> Execute tasks with idempotency and retry support.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, registry: Optional[TaskRegistry] = None, max_retries: int = 3)` | 144 |  |
| ⚙️ | `execute` | `(self, func: Callable, *args: Any, **kwargs: Any) -> Any` | 148 | Execute a function as an idempotent task. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_task_registry` | `() -> TaskRegistry` | 81 |  |
| 🔧 | `idempotent_task` | `(registry: Optional[TaskRegistry] = None, cache_result: bool = True) -> Callable` | 85 | Decorator to make a function idempotent for workflow replay. |

---

## `argus/durable/workflow.py`

### 🏛️ `class WorkflowStatus(str, Enum)`  <sub>line 27</sub>

> Workflow execution status.  

### 🏛️ `class WorkflowRun()`  <sub>line 38</sub>

> Decorators: `@dataclass`  
> A single workflow execution run.  

### 🏛️ `class DurableWorkflow()`  <sub>line 51</sub>

> Durable workflow with checkpointing and resume capability.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, thread_id: Optional[str] = None, config: Optional[DurableConfig] = None, checkpointer: Optional[BaseCheckpointer] = None)` | 70 |  |
| ⚙️ | `step` | `(self, name: str, func: Callable, *args: Any, **kwargs: Any) -> Any` | 84 | Execute a workflow step with automatic tracking. |
| ⚙️ | `checkpoint` | `(self, description: str = '') -> str` | 105 | Create a checkpoint of current state. |
| ⚙️ | `resume` | `(self, checkpoint_id: Optional[str] = None) -> bool` | 121 | Resume workflow from checkpoint. |
| ⚙️ | `rollback` | `(self, steps: int = 1) -> bool` | 150 | Rollback to a previous checkpoint. |
| ⚙️ | `start_run` | `(self, total_steps: int = 0) -> WorkflowRun` | 159 | Start a new workflow run. |
| ⚙️ | `complete_run` | `(self, error: Optional[str] = None) -> WorkflowRun` | 171 | Complete the current workflow run. |
| ⚙️ | `get_current_run` | `(self) -> Optional[WorkflowRun]` | 181 |  |
| ⚙️ | `list_checkpoints` | `(self, limit: int = 10) -> List[Checkpoint]` | 184 |  |

### 🏛️ `class DurableDebateWorkflow(DurableWorkflow)`  <sub>line 188</sub>

> Durable workflow specialized for debate orchestration.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, orchestrator: Optional['RDCOrchestrator'] = None, **kwargs: Any)` | 191 |  |
| ⚙️ | `initialize_debate` | `(self, proposition: str, prior: float = 0.5) -> DebateState` | 195 | Initialize a new debate. |
| ⚙️ | `run_round` | `(self, round_num: int) -> dict[str, Any]` | 206 | Execute a debate round as a workflow step. |
| ⚙️ | `execute_round` | `() -> dict[str, Any]` | 208 |  |

---

## `argus/evaluation/benchmarks/base.py`

### 🏛️ `class BenchmarkStatus(Enum)`  <sub>line 24</sub>

> Status of a benchmark run.  

### 🏛️ `class BenchmarkConfig()`  <sub>line 34</sub>

> Decorators: `@dataclass`  
> Configuration for benchmark execution.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 54 |  |

### 🏛️ `class SampleResult()`  <sub>line 60</sub>

> Decorators: `@dataclass`  
> Result for a single benchmark sample.  

### 🏛️ `class BenchmarkResult()`  <sub>line 86</sub>

> Decorators: `@dataclass`  
> Aggregated result of a benchmark run.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `duration_seconds` | `(self) -> float` | 121 | Total benchmark duration. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 127 | Convert to dictionary for serialization. |

### 🏛️ `class Benchmark(ABC)`  <sub>line 146</sub>

> Abstract base class for benchmarks.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, name: str, description: str = '')` | 162 | Initialize benchmark. |
| ⚙️ `@abstractmethod` | `evaluate_sample` | `(self, sample: dict[str, Any], llm: Any, config: BenchmarkConfig) -> SampleResult` | 173 | Evaluate a single sample. |
| ⚙️ | `run` | `(self, dataset: 'pd.DataFrame', llm: Any, config: Optional[BenchmarkConfig] = None) -> BenchmarkResult` | 191 | Run benchmark on dataset. |
| ⚙️ | `_aggregate_scores` | `(self, sample_results: list[SampleResult]) -> dict[str, float]` | 269 | Aggregate scores across samples. |

---

## `argus/evaluation/benchmarks/debate_quality.py`

### 🏛️ `class DebateQualityBenchmark(Benchmark)`  <sub>line 23</sub>

> Benchmark for evaluating debate quality.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 38 |  |
| ⚙️ | `evaluate_sample` | `(self, sample: dict[str, Any], llm: Any, config: BenchmarkConfig) -> SampleResult` | 44 | Evaluate a single proposition using ARGUS debate. |
| ⚙️ | `_check_verdict_match` | `(self, predicted: str, expected: str) -> bool` | 125 | Check if predicted verdict matches expected. |
| ⚙️ | `normalize` | `(verdict: str) -> str` | 142 |  |

---

## `argus/evaluation/benchmarks/evidence_analysis.py`

### 🏛️ `class EvidenceAnalysisBenchmark(Benchmark)`  <sub>line 22</sub>

> Benchmark for evaluating evidence handling quality.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 36 |  |
| ⚙️ | `evaluate_sample` | `(self, sample: dict[str, Any], llm: Any, config: BenchmarkConfig) -> SampleResult` | 42 | Evaluate evidence handling for a single proposition. |
| ⚙️ | `_analyze_evidence` | `(self, result: Any) -> dict[str, Any]` | 125 | Analyze evidence quality from debate result. |

---

## `argus/evaluation/benchmarks/reasoning_depth.py`

### 🏛️ `class ReasoningDepthBenchmark(Benchmark)`  <sub>line 23</sub>

> Benchmark for evaluating reasoning depth and sophistication.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 37 |  |
| ⚙️ | `evaluate_sample` | `(self, sample: dict[str, Any], llm: Any, config: BenchmarkConfig) -> SampleResult` | 43 | Evaluate reasoning depth for a single proposition. |
| ⚙️ | `_analyze_reasoning` | `(self, result: Any) -> dict[str, Any]` | 126 | Analyze reasoning depth from debate result. |

---

## `argus/evaluation/datasets/global_benchmarks.py`

### 🏛️ `class HFBenchmarkInfo()`  <sub>line 41</sub>

> Decorators: `@dataclass`  
> Information about a Hugging Face benchmark dataset.  

### 🏛️ `class HuggingFaceDatasetLoader()`  <sub>line 205</sub>

> Loader for global benchmark datasets using Hugging Face datasets library.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, cache_dir: Optional[Path] = None)` | 217 | Initialize loader. |
| ⚙️ | `_check_datasets_available` | `(self) -> bool` | 226 | Check if huggingface datasets library is available. |
| ⚙️ | `list_benchmarks` | `(self) -> list[str]` | 238 | List available benchmarks. |
| ⚙️ | `get_benchmark_info` | `(self, name: str) -> Optional[HFBenchmarkInfo]` | 242 | Get information about a benchmark. |
| ⚙️ | `load` | `(self, name: str, max_samples: int = 1000, split: Optional[str] = None, streaming: bool = False) -> 'pd.DataFrame'` | 253 | Load a benchmark dataset from Hugging Face. |
| ⚙️ | `_convert_to_argus_format` | `(self, hf_dataset: Any, benchmark: HFBenchmarkInfo) -> 'pd.DataFrame'` | 335 | Convert Hugging Face dataset to ARGUS evaluation format. |
| ⚙️ | `_get_field` | `(self, sample: dict, field_name: str, default: Any = '') -> Any` | 417 | Safely get field from sample. |
| ⚙️ | `to_csv` | `(self, name: str, output_path: Path, max_samples: int = 1000) -> Path` | 440 | Load benchmark and save as CSV. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `list_global_benchmarks` | `() -> list[str]` | 468 | List all available global benchmarks. |
| 🔧 | `load_global_benchmark` | `(name: str, max_samples: int = 1000, split: Optional[str] = None) -> 'pd.DataFrame'` | 477 | Load a global benchmark from Hugging Face as DataFrame. |
| 🔧 | `get_benchmark_info` | `(name: str) -> Optional[HFBenchmarkInfo]` | 533 | Get detailed information about a benchmark. |
| 🔧 | `download_all_benchmarks` | `(output_dir: Path, max_samples_per_benchmark: int = 1000) -> dict[str, Path]` | 545 | Download all benchmarks as CSV files. |

---

## `argus/evaluation/datasets/loader.py`

### 🏛️ `class DatasetInfo()`  <sub>line 22</sub>

> Decorators: `@dataclass`  
> Information about a dataset.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 40 |  |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `list_datasets` | `() -> list[str]` | 110 | List all available dataset names. |
| 🔧 | `get_dataset_path` | `(name: str) -> Path` | 119 | Get the file path for a dataset. |
| 🔧 | `load_dataset` | `(name: str, max_samples: int = -1) -> 'pd.DataFrame'` | 138 | Load a dataset by name. |
| 🔧 | `validate_dataset` | `(name: str) -> tuple[bool, list[str]]` | 172 | Validate a dataset for completeness and correctness. |
| 🔧 | `validate_all` | `() -> dict[str, tuple[bool, list[str]]]` | 231 | Validate all registered datasets. |

---

## `argus/evaluation/runner/benchmark_runner.py`

### 🏛️ `class RunConfig()`  <sub>line 32</sub>

> Decorators: `@dataclass`  
> Configuration for benchmark runner.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 54 |  |

### 🏛️ `class BenchmarkRunner()`  <sub>line 58</sub>

> Runs benchmarks and collects results.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[RunConfig] = None)` | 71 | Initialize runner. |
| ⚙️ | `run` | `(self, llm: Any, config: Optional[RunConfig] = None) -> list[BenchmarkResult]` | 83 | Run all configured benchmarks. |
| ⚙️ | `_get_benchmarks` | `(self, names: list[str]) -> list[Benchmark]` | 145 | Get benchmark instances by name. |
| ⚙️ | `_dry_run` | `(self, benchmark: Benchmark, dataset_name: str, config: BenchmarkConfig) -> BenchmarkResult` | 175 | Perform dry run without actual LLM calls. |
| ⚙️ | `_save_result` | `(self, result: BenchmarkResult)` | 217 | Save individual result to disk. |
| ⚙️ | `_save_summary` | `(self)` | 231 | Save summary of all results. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `main` | `()` | 246 | CLI entry point for benchmark runner. |

---

## `argus/evaluation/runner/report_generator.py`

### 🏛️ `class ReportGenerator()`  <sub>line 21</sub>

> Generates evaluation reports in various formats.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, results: list[BenchmarkResult])` | 32 | Initialize with results. |
| ⚙️ | `generate_markdown` | `(self, output_path: Optional[Path] = None) -> str` | 41 | Generate markdown report. |
| ⚙️ | `generate_json` | `(self, output_path: Optional[Path] = None) -> str` | 114 | Generate JSON report. |
| ⚙️ | `generate_html` | `(self, output_path: Optional[Path] = None) -> str` | 139 | Generate HTML report. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_report` | `(results: list[BenchmarkResult], output_path: Path, format: str = 'markdown') -> str` | 243 | Convenience function to generate report. |

---

## `argus/evaluation/runner/results_aggregator.py`

### 🏛️ `class AggregatedMetrics()`  <sub>line 23</sub>

> Decorators: `@dataclass`  
> Aggregated metrics across multiple runs.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 41 |  |

### 🏛️ `class ResultsAggregator()`  <sub>line 52</sub>

> Aggregates results from multiple benchmark runs.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 61 | Initialize aggregator. |
| ⚙️ | `add_result` | `(self, result: BenchmarkResult)` | 65 | Add a single result. |
| ⚙️ | `add_results` | `(self, results: list[BenchmarkResult])` | 73 | Add multiple results. |
| ⚙️ | `load_from_directory` | `(self, directory: Path)` | 81 | Load results from saved JSON files. |
| ⚙️ | `summarize` | `(self) -> dict[str, Any]` | 105 | Generate summary statistics. |
| ⚙️ | `compare_runs` | `(self, baseline_name: str, comparison_name: str) -> dict[str, Any]` | 164 | Compare two benchmark runs. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `aggregate_results` | `(results: list[BenchmarkResult]) -> dict[str, Any]` | 197 | Convenience function to aggregate results. |

---

## `argus/evaluation/scoring/aggregate.py`

### 🏛️ `class CompositeScore()`  <sub>line 22</sub>

> Decorators: `@dataclass`  
> Weighted composite of multiple metric scores.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 36 |  |
| ⚙️ | `_compute` | `(self) -> float` | 45 | Compute weighted composite. |
| ⚙️ | `with_weights` | `(self, weights: dict[str, float]) -> 'CompositeScore'` | 60 | Create new composite with different weights. |

### 🏛️ `class ScoreComparison()`  <sub>line 66</sub>

> Decorators: `@dataclass`  
> Comparison between two score cards.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 82 |  |
| ⚙️ | `_compute_differences` | `(self)` | 85 | Compute score differences. |
| ⚙️ `@property` | `composite_change` | `(self) -> float` | 110 | Change in composite score. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 114 | Convert to dictionary. |
| ⚙️ | `__str__` | `(self) -> str` | 125 | Format as readable string. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_score_report` | `(score_cards: list[ScoreCard], output_path: Optional[Path] = None, format: str = 'markdown') -> str` | 144 | Generate a formatted report from multiple score cards. |
| 🔧 | `_generate_markdown_report` | `(score_cards: list[ScoreCard]) -> str` | 179 | Generate markdown format report. |
| 🔧 | `_generate_html_report` | `(score_cards: list[ScoreCard]) -> str` | 220 | Generate HTML format report. |

---

## `argus/evaluation/scoring/metrics.py`

### 🏛️ `class MetricDefinition()`  <sub>line 36</sub>

> Decorators: `@dataclass`  
> Definition of a scoring metric.  

### 🏛️ `class ScoreCard()`  <sub>line 769</sub>

> Decorators: `@dataclass`  
> Complete score card for a debate evaluation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 783 |  |
| ⚙️ | `_compute_composite` | `(self) -> float` | 787 | Compute weighted composite score. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 800 | Convert to dictionary. |
| ⚙️ | `__str__` | `(self) -> str` | 809 | Format as readable string. |
| ⚙️ `@classmethod` | `from_result` | `(cls, result: dict[str, Any]) -> 'ScoreCard'` | 819 | Create ScoreCard from debate result. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `compute_arcis` | `(result: dict[str, Any]) -> float` | 135 | Compute ARCIS - Argus Reasoning Coherence Index Score. |
| 🔧 | `compute_evid_q` | `(result: dict[str, Any]) -> float` | 241 | Compute EVID-Q - Evidence Quality Quotient. |
| 🔧 | `compute_dialec` | `(result: dict[str, Any]) -> float` | 309 | Compute DIALEC - Dialectical Depth Evaluation Coefficient. |
| 🔧 | `compute_rebut_f` | `(result: dict[str, Any]) -> float` | 380 | Compute REBUT-F - Rebuttal Effectiveness Factor. |
| 🔧 | `compute_conv_s` | `(result: dict[str, Any]) -> float` | 451 | Compute CONV-S - Convergence Stability Score. |
| 🔧 | `compute_prov_i` | `(result: dict[str, Any]) -> float` | 534 | Compute PROV-I - Provenance Integrity Index. |
| 🔧 | `compute_calib_m` | `(result: dict[str, Any]) -> float` | 609 | Compute CALIB-M - Calibration Matrix Score. |
| 🔧 | `compute_eig_u` | `(result: dict[str, Any]) -> float` | 674 | Compute EIG-U - Expected Information Gain Utilization. |
| 🔧 | `compute_all_scores` | `(result: dict[str, Any]) -> dict[str, float]` | 746 | Compute all 8 ARGUS novel metrics for a debate result. |

---

## `argus/evaluation/scoring/standard_metrics.py`

### 🏛️ `class StandardMetricsResult()`  <sub>line 365</sub>

> Decorators: `@dataclass`  
> Results from standard metrics computation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, float]` | 379 |  |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `compute_accuracy` | `(predictions: Sequence[str], ground_truths: Sequence[str]) -> float` | 27 | Compute classification accuracy. |
| 🔧 | `compute_precision_recall_f1` | `(predictions: Sequence[str], ground_truths: Sequence[str], positive_label: str = 'supported') -> dict[str, float]` | 51 | Compute precision, recall, and F1 score. |
| 🔧 | `compute_macro_f1` | `(predictions: Sequence[str], ground_truths: Sequence[str], labels: Optional[Sequence[str]] = None) -> float` | 91 | Compute macro-averaged F1 score across all classes. |
| 🔧 | `compute_brier_score` | `(confidences: Sequence[float], outcomes: Sequence[int]) -> float` | 122 | Compute Brier Score for probability calibration. |
| 🔧 | `compute_ece` | `(confidences: Sequence[float], outcomes: Sequence[int], num_bins: int = 10) -> tuple[float, dict[str, Any]]` | 148 | Compute Expected Calibration Error (ECE). |
| 🔧 | `compute_mce` | `(confidences: Sequence[float], outcomes: Sequence[int], num_bins: int = 10) -> float` | 200 | Compute Maximum Calibration Error (MCE). |
| 🔧 | `compute_cross_entropy` | `(confidences: Sequence[float], outcomes: Sequence[int], epsilon: float = 1e-15) -> float` | 231 | Compute binary cross-entropy loss. |
| 🔧 | `compute_log_loss` | `(confidences: Sequence[float], outcomes: Sequence[int]) -> float` | 258 | Compute log loss (same as cross-entropy for binary classification). |
| 🔧 | `compute_argument_coverage` | `(evidence_count: int, expected_evidence: int = 5) -> float` | 279 | Compute argument coverage score. |
| 🔧 | `compute_dialectical_balance` | `(support_count: int, attack_count: int) -> float` | 300 | Compute dialectical balance score. |
| 🔧 | `compute_verdict_confidence_correlation` | `(posteriors: Sequence[float], correctness: Sequence[bool]) -> float` | 327 | Compute correlation between posterior confidence and correctness. |
| 🔧 | `compute_all_standard_metrics` | `(predictions: Sequence[str], ground_truths: Sequence[str], confidences: Optional[Sequence[float]] = None, outcomes: Optional[Sequence[int]] = None, evidence_count: int = 0, support_count: int = 0, attack_count: int = 0) -> StandardMetricsResult` | 395 | Compute all standard metrics in one call. |

---

## `argus/hitl/callbacks.py`

### 🏛️ `class FeedbackType(str, Enum)`  <sub>line 19</sub>

> Types of feedback.  

### 🏛️ `class Feedback()`  <sub>line 28</sub>

> Decorators: `@dataclass`  
> Feedback from a human reviewer.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 38 |  |

### 🏛️ `class BaseCallback(ABC)`  <sub>line 50</sub>

> Abstract base class for HITL callbacks.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 53 |  |
| ⚙️ `@abstractmethod` | `collect` | `(self, **kwargs: Any) -> Feedback` | 57 |  |
| ⚙️ | `get_feedback` | `(self, limit: int = 100) -> list[Feedback]` | 60 |  |

### 🏛️ `class FeedbackCallback(BaseCallback)`  <sub>line 64</sub>

> General feedback collection callback.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, on_feedback: Optional[Callable[[Feedback], None]] = None)` | 67 |  |
| ⚙️ | `collect` | `(self, feedback_type: FeedbackType, content: Any, agent_name: Optional[str] = None, action_name: Optional[str] = None, **kwargs: Any) -> Feedback` | 71 |  |

### 🏛️ `class RatingCallback(BaseCallback)`  <sub>line 95</sub>

> Collect numeric ratings (1-5).  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, min_rating: int = 1, max_rating: int = 5)` | 98 |  |
| ⚙️ | `collect` | `(self, rating: int, agent_name: Optional[str] = None, action_name: Optional[str] = None, comment: Optional[str] = None, **kwargs: Any) -> Feedback` | 103 |  |
| ⚙️ | `get_average_rating` | `(self, action_name: Optional[str] = None) -> float` | 126 |  |

### 🏛️ `class AnnotationCallback(BaseCallback)`  <sub>line 133</sub>

> Collect text annotations on agent outputs.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `collect` | `(self, annotation: str, target_id: str, agent_name: Optional[str] = None, action_name: Optional[str] = None, **kwargs: Any) -> Feedback` | 136 |  |

### 🏛️ `class CorrectionCallback(BaseCallback)`  <sub>line 158</sub>

> Record corrections to agent outputs for learning.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `collect` | `(self, original: str, corrected: str, agent_name: Optional[str] = None, action_name: Optional[str] = None, reason: Optional[str] = None, **kwargs: Any) -> Feedback` | 161 |  |
| ⚙️ | `get_corrections_for_agent` | `(self, agent_name: str) -> list[Feedback]` | 183 |  |

### 🏛️ `class CallbackManager()`  <sub>line 187</sub>

> Manages multiple callback types.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 190 |  |
| ⚙️ | `get_all_feedback` | `(self, limit: int = 100) -> list[Feedback]` | 196 |  |

---

## `argus/hitl/config.py`

### 🏛️ `class ApprovalMode(str, Enum)`  <sub>line 19</sub>

> Approval mode for agent actions.  

### 🏛️ `class InterruptionPoint(str, Enum)`  <sub>line 29</sub>

> Points where workflow can be interrupted for human input.  

### 🏛️ `class SensitivityLevel(str, Enum)`  <sub>line 45</sub>

> Sensitivity level of actions for approval filtering.  

### 🏛️ `class FeedbackType(str, Enum)`  <sub>line 56</sub>

> Types of feedback that can be collected from humans.  

### 🏛️ `class HITLCallbackConfig(BaseModel)`  <sub>line 65</sub>

> Configuration for HITL callbacks.  

### 🏛️ `class HITLConfig(BaseModel)`  <sub>line 89</sub>

> Main configuration for Human-in-the-Loop functionality.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `should_require_approval` | `(self, sensitivity: SensitivityLevel) -> bool` | 163 | Check if an action with given sensitivity requires approval. |
| ⚙️ | `should_interrupt_at` | `(self, point: InterruptionPoint) -> bool` | 194 | Check if workflow should be interrupted at given point. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_default_hitl_config` | `() -> HITLConfig` | 209 | Get default HITL configuration. |

---

## `argus/hitl/handlers.py`

### 🏛️ `class HandlerResult()`  <sub>line 22</sub>

> Decorators: `@dataclass`  
> Result from a decision handler.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@classmethod` | `approved` | `(cls, message: str = 'Action approved') -> 'HandlerResult'` | 32 |  |
| ⚙️ `@classmethod` | `rejected` | `(cls, message: str = 'Action rejected') -> 'HandlerResult'` | 36 |  |
| ⚙️ `@classmethod` | `modified` | `(cls, modified_args: dict[str, Any], message: str = 'Action modified') -> 'HandlerResult'` | 40 |  |

### 🏛️ `class BaseHandler(ABC)`  <sub>line 44</sub>

> Abstract base class for HITL decision handlers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[HITLConfig] = None)` | 47 |  |
| ⚙️ `@abstractmethod` | `handle` | `(self, request: InterruptRequest) -> HandlerResult` | 53 |  |
| ⚙️ | `record` | `(self, request: InterruptRequest, result: HandlerResult) -> None` | 56 |  |

### 🏛️ `class ApprovalHandler(BaseHandler)`  <sub>line 66</sub>

> Handler for approved actions.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[HITLConfig] = None, pre_execute_hook: Optional[Callable] = None)` | 69 |  |
| ⚙️ | `handle` | `(self, request: InterruptRequest) -> HandlerResult` | 73 |  |

### 🏛️ `class RejectionHandler(BaseHandler)`  <sub>line 84</sub>

> Handler for rejected actions.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[HITLConfig] = None, require_reason: bool = False)` | 87 |  |
| ⚙️ | `handle` | `(self, request: InterruptRequest, reason: Optional[str] = None) -> HandlerResult` | 92 |  |

### 🏛️ `class ModificationHandler(BaseHandler)`  <sub>line 104</sub>

> Handler for modified actions.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[HITLConfig] = None, allowed_fields: Optional[set[str]] = None)` | 107 |  |
| ⚙️ | `handle` | `(self, request: InterruptRequest, modified_args: Optional[dict[str, Any]] = None) -> HandlerResult` | 111 |  |

### 🏛️ `class EscalationHandler(BaseHandler)`  <sub>line 121</sub>

> Handler for escalated actions.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[HITLConfig] = None, escalation_callback: Optional[Callable] = None)` | 124 |  |
| ⚙️ | `handle` | `(self, request: InterruptRequest, reason: str = '') -> HandlerResult` | 129 |  |

### 🏛️ `class DecisionRouter()`  <sub>line 143</sub>

> Routes interrupt requests to appropriate handlers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[HITLConfig] = None)` | 146 |  |
| ⚙️ | `route` | `(self, request: InterruptRequest, decision: InterruptStatus, **kwargs: Any) -> HandlerResult` | 153 |  |

---

## `argus/hitl/middleware.py`

### 🏛️ `class InterruptStatus(str, Enum)`  <sub>line 36</sub>

> Status of an interrupt request.  

### 🏛️ `class InterruptRequest()`  <sub>line 48</sub>

> Decorators: `@dataclass`  
> Represents a pending action awaiting human decision.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 80 | Convert to dictionary for serialization. |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'InterruptRequest'` | 91 | Create from dictionary. |

### 🏛️ `class MiddlewareState()`  <sub>line 104</sub>

> Decorators: `@dataclass`  
> State preserved during workflow interruption.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_json` | `(self) -> str` | 116 | Serialize to JSON string. |
| ⚙️ `@classmethod` | `from_json` | `(cls, json_str: str) -> 'MiddlewareState'` | 129 | Deserialize from JSON string. |

### 🏛️ `class HITLMiddleware()`  <sub>line 137</sub>

> Middleware for human-in-the-loop intervention.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[HITLConfig] = None, response_handler: Optional[Callable[[InterruptRequest], InterruptStatus]] = None)` | 156 | Initialize middleware. |
| ⚙️ | `should_intercept` | `(self, action_name: str, point: InterruptionPoint, sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM) -> bool` | 177 | Check if an action should be intercepted at given point. |
| ⚙️ | `create_interrupt` | `(self, action_name: str, action_args: dict[str, Any], point: InterruptionPoint, sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM, context: Optional[dict[str, Any]] = None) -> InterruptRequest` | 201 | Create an interrupt request for human review. |
| ⚙️ | `save_state` | `(self, session_id: str, request: InterruptRequest, context: Optional[dict[str, Any]] = None, agent_state: Optional[dict[str, Any]] = None, graph_state: Optional[dict[str, Any]] = None) -> MiddlewareState` | 245 | Save workflow state during interruption. |
| ⚙️ | `restore_state` | `(self, session_id: str) -> Optional[MiddlewareState]` | 280 | Restore workflow state after interruption. |
| ⚙️ | `submit_response` | `(self, request_id: str, status: InterruptStatus, response: Optional[str] = None, modified_action: Optional[dict[str, Any]] = None) -> bool` | 301 | Submit human response to an interrupt request. |
| ⚙️ | `get_pending_requests` | `(self) -> list[InterruptRequest]` | 347 | Get all pending interrupt requests. |
| ⚙️ | `get_request` | `(self, request_id: str) -> Optional[InterruptRequest]` | 356 | Get a specific interrupt request. |
| ⚙️ | `check_timeout` | `(self, request_id: str) -> bool` | 368 | Check if a request has timed out. |
| ⚙️ | `get_decision_history` | `(self, limit: int = 100, action_name: Optional[str] = None) -> list[dict[str, Any]]` | 393 | Get history of human decisions. |
| ⚙️ | `clear_pending` | `(self) -> int` | 415 | Clear all pending requests. |

### 🏛️ `class ToolInterceptor()`  <sub>line 429</sub>

> Interceptor that wraps tool execution with HITL middleware.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, middleware: HITLMiddleware)` | 443 | Initialize interceptor. |
| ⚙️ | `wrap` | `(self, sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM, point: InterruptionPoint = InterruptionPoint.BEFORE_TOOL_CALL) -> Callable` | 451 | Decorator to wrap a function with HITL interception. |
| ⚙️ | `decorator` | `(func: Callable) -> Callable` | 465 |  |
| ⚙️ | `wrapper` | `(*args: Any, **kwargs: Any) -> Any` | 466 |  |

---

## `argus/knowledge/chunking.py`

### 🏛️ `class ChunkingStrategy(str, Enum)`  <sub>line 21</sub>

> Chunking strategy options.  

### 🏛️ `class ChunkingConfig()`  <sub>line 31</sub>

> Decorators: `@dataclass`  
> Configuration for chunking.  

### 🏛️ `class Chunker()`  <sub>line 49</sub>

> Document chunker.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100, max_chunk_size: int = 1000, strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE)` | 63 | Initialize chunker. |
| ⚙️ | `chunk` | `(self, document: Document) -> list[Chunk]` | 90 | Chunk a document. |
| ⚙️ | `_estimate_tokens` | `(self, text: str) -> int` | 111 | Estimate token count for text. |
| ⚙️ | `_create_chunk` | `(self, doc_id: str, text: str, start_char: int, end_char: int, chunk_index: int) -> Chunk` | 115 | Create a Chunk object. |
| ⚙️ | `_chunk_fixed` | `(self, document: Document) -> list[Chunk]` | 133 | Chunk by fixed size with overlap. |
| ⚙️ | `_chunk_sentences` | `(self, document: Document) -> list[Chunk]` | 171 | Chunk by sentence boundaries. |
| ⚙️ | `_chunk_paragraphs` | `(self, document: Document) -> list[Chunk]` | 216 | Chunk by paragraph boundaries. |
| ⚙️ | `_chunk_recursive` | `(self, document: Document) -> list[Chunk]` | 262 | Recursively chunk using multiple separators. |
| ⚙️ | `split_text` | `(txt: str, start_offset: int, sep_idx: int) -> list[tuple[str, int, int]]` | 283 | Recursively split text and return (text, start, end) tuples. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `chunk_document` | `(document: Document, chunk_size: int = 512, chunk_overlap: int = 50, strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE) -> list[Chunk]` | 361 | Convenience function to chunk a document. |

---

## `argus/knowledge/connectors/arxiv.py`

### 🏛️ `class ArxivPaper()`  <sub>line 35</sub>

> Decorators: `@dataclass`  
> Represents an arXiv paper with metadata.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_document` | `(self) -> Document` | 50 | Convert to ARGUS Document. |

### 🏛️ `class ArxivConnectorConfig(ConnectorConfig)`  <sub>line 81</sub>

> Configuration for arXiv connector.  

### 🏛️ `class ArxivConnector(BaseConnector)`  <sub>line 103</sub>

> Connector for fetching papers from arXiv.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ArxivConnectorConfig] = None)` | 144 | Initialize arXiv connector. |
| ⚙️ `@property` | `arxiv_config` | `(self) -> ArxivConnectorConfig` | 154 | Get typed config. |
| ⚙️ | `_get_client` | `(self)` | 158 | Get or create arxiv client. |
| ⚙️ | `_build_query` | `(self, query: str, categories: Optional[list[str]] = None, authors: Optional[list[str]] = None, date_from: Optional[str] = None, date_to: Optional[str] = None) -> str` | 174 | Build arXiv search query. |
| ⚙️ | `_parse_paper` | `(self, result: Any) -> ArxivPaper` | 229 | Parse arxiv result into ArxivPaper. |
| ⚙️ | `fetch` | `(self, query: str, max_results: int = 10, **kwargs: Any) -> ConnectorResult` | 256 | Fetch papers from arXiv. |
| ⚙️ | `fetch_by_id` | `(self, arxiv_id: str) -> ConnectorResult` | 360 | Fetch a specific paper by arXiv ID. |
| ⚙️ | `fetch_by_category` | `(self, categories: list[str], max_results: int = 10, sort_by: str = 'submittedDate') -> ConnectorResult` | 373 | Fetch recent papers from specified categories. |
| ⚙️ | `test_connection` | `(self) -> bool` | 398 | Test connection to arXiv API. |
| ⚙️ | `validate_config` | `(self) -> Optional[str]` | 410 | Validate connector configuration. |

---

## `argus/knowledge/connectors/base.py`

### 🏛️ `class ConnectorConfig(BaseModel)`  <sub>line 37</sub>

> Configuration for a connector.  

### 🏛️ `class ConnectorResult()`  <sub>line 70</sub>

> Decorators: `@dataclass`  
> Result from a connector fetch operation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 86 |  |
| ⚙️ `@classmethod` | `from_error` | `(cls, error: str) -> 'ConnectorResult'` | 97 | Create a failed result. |

### 🏛️ `class BaseConnector(ABC)`  <sub>line 102</sub>

> Abstract base class for external data connectors.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ConnectorConfig] = None)` | 126 | Initialize the connector. |
| ⚙️ `@abstractmethod` | `fetch` | `(self, query: str, max_results: int = 10, **kwargs: Any) -> ConnectorResult` | 139 | Fetch documents matching the query. |
| ⚙️ | `test_connection` | `(self) -> bool` | 159 | Test if the connector can connect to its data source. |
| ⚙️ | `validate_config` | `(self) -> Optional[str]` | 169 | Validate connector configuration. |
| ⚙️ | `_rate_limit_wait` | `(self) -> None` | 179 | Wait to respect rate limit. |
| ⚙️ | `__call__` | `(self, query: str, max_results: int = 10, **kwargs: Any) -> ConnectorResult` | 192 | Fetch documents (callable interface). |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 235 | Get connector schema/metadata. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 248 | Get connector statistics. |
| ⚙️ | `__repr__` | `(self) -> str` | 260 |  |

### 🏛️ `class ConnectorRegistry()`  <sub>line 264</sub>

> Registry for managing connectors.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 275 | Initialize the registry. |
| ⚙️ | `register` | `(self, connector: BaseConnector) -> None` | 281 | Register a connector. |
| ⚙️ | `unregister` | `(self, name: str) -> bool` | 299 | Unregister a connector. |
| ⚙️ | `get` | `(self, name: str) -> Optional[BaseConnector]` | 314 | Get a connector by name. |
| ⚙️ | `list_all` | `(self) -> list[str]` | 326 | Get all registered connector names. |
| ⚙️ | `get_all` | `(self) -> list[BaseConnector]` | 335 | Get all registered connectors. |
| ⚙️ | `fetch_from_all` | `(self, query: str, max_results_per_connector: int = 5) -> dict[str, ConnectorResult]` | 344 | Fetch from all registered connectors. |
| ⚙️ | `__len__` | `(self) -> int` | 366 |  |
| ⚙️ | `__contains__` | `(self, name: str) -> bool` | 369 |  |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_default_registry` | `() -> ConnectorRegistry` | 381 | Get the default connector registry. |
| 🔧 | `register_connector` | `(connector: BaseConnector) -> None` | 395 | Register a connector with the default registry. |

---

## `argus/knowledge/connectors/crossref.py`

### 🏛️ `class CrossRefWork()`  <sub>line 36</sub>

> Decorators: `@dataclass`  
> Represents a CrossRef work (publication) with metadata.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `author_names` | `(self) -> list[str]` | 58 | Get formatted author names. |
| ⚙️ `@property` | `citation_string` | `(self) -> str` | 73 | Generate a citation string. |
| ⚙️ | `to_document` | `(self) -> Document` | 111 | Convert to ARGUS Document. |

### 🏛️ `class CrossRefConnectorConfig(ConnectorConfig)`  <sub>line 153</sub>

> Configuration for CrossRef connector.  

### 🏛️ `class CrossRefConnector(BaseConnector)`  <sub>line 175</sub>

> Connector for fetching citation metadata from CrossRef.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[CrossRefConnectorConfig] = None)` | 200 | Initialize CrossRef connector. |
| ⚙️ `@property` | `crossref_config` | `(self) -> CrossRefConnectorConfig` | 210 | Get typed config. |
| ⚙️ | `_get_session` | `(self)` | 214 | Get or create HTTP session. |
| ⚙️ | `_make_request` | `(self, endpoint: str, params: Optional[dict] = None) -> dict` | 233 | Make a request to CrossRef API. |
| ⚙️ | `_parse_date` | `(self, date_parts: Optional[list]) -> Optional[datetime]` | 275 | Parse CrossRef date-parts into datetime. |
| ⚙️ | `_parse_work` | `(self, item: dict) -> CrossRefWork` | 297 | Parse CrossRef work JSON into CrossRefWork. |
| ⚙️ | `fetch` | `(self, query: str, max_results: int = 10, **kwargs: Any) -> ConnectorResult` | 350 | Fetch works from CrossRef. |
| ⚙️ | `fetch_by_doi` | `(self, doi: str) -> ConnectorResult` | 422 | Fetch a specific work by DOI. |
| ⚙️ | `fetch_references` | `(self, doi: str, max_results: int = 50) -> ConnectorResult` | 463 | Fetch references for a work. |
| ⚙️ | `fetch_citing_works` | `(self, doi: str, max_results: int = 20) -> ConnectorResult` | 536 | Fetch works that cite a given DOI. |
| ⚙️ | `test_connection` | `(self) -> bool` | 558 | Test connection to CrossRef API. |
| ⚙️ | `close` | `(self) -> None` | 570 | Close HTTP session. |

---

## `argus/knowledge/connectors/web.py`

### 🏛️ `class RobotRule()`  <sub>line 40</sub>

> Decorators: `@dataclass`  
> A single rule from robots.txt.  

### 🏛️ `class RobotsTxtRules()`  <sub>line 47</sub>

> Decorators: `@dataclass`  
> Parsed robots.txt rules for a user-agent.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `is_allowed` | `(self, path: str) -> bool` | 54 | Check if a path is allowed by these rules. |

### 🏛️ `class RobotsTxtParser()`  <sub>line 106</sub>

> Parser for robots.txt files with caching.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, cache_ttl: int = 3600)` | 120 | Initialize the parser. |
| ⚙️ | `parse` | `(self, content: str, user_agent: str = '*') -> RobotsTxtRules` | 129 | Parse robots.txt content. |
| ⚙️ | `fetch_and_parse` | `(self, base_url: str, user_agent: str = '*', session: Any = None, timeout: float = 10.0) -> RobotsTxtRules` | 212 | Fetch and parse robots.txt from a URL. |
| ⚙️ | `is_allowed` | `(self, url: str, user_agent: str = '*', session: Any = None, timeout: float = 10.0) -> bool` | 266 | Check if a URL is allowed by robots.txt. |
| ⚙️ | `clear_cache` | `(self) -> None` | 293 | Clear the robots.txt cache. |

### 🏛️ `class WebConnectorConfig(ConnectorConfig)`  <sub>line 302</sub>

> Configuration for web connector.  

### 🏛️ `class WebConnector(BaseConnector)`  <sub>line 345</sub>

> Connector for fetching web content with robots.txt compliance.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[WebConnectorConfig] = None)` | 366 | Initialize web connector. |
| ⚙️ `@property` | `web_config` | `(self) -> WebConnectorConfig` | 379 | Get typed config. |
| ⚙️ | `_get_session` | `(self)` | 383 | Get or create HTTP session. |
| ⚙️ | `_check_robots_txt` | `(self, url: str) -> bool` | 397 | Check if URL is allowed by robots.txt. |
| ⚙️ | `_get_crawl_delay` | `(self, url: str) -> Optional[float]` | 416 | Get crawl delay from robots.txt. |
| ⚙️ | `fetch` | `(self, query: str, max_results: int = 1, **kwargs: Any) -> ConnectorResult` | 436 | Fetch content from URL with robots.txt compliance. |
| ⚙️ | `fetch_multiple` | `(self, urls: list[str], max_concurrent: int = 5, **kwargs: Any) -> list[ConnectorResult]` | 548 | Fetch multiple URLs. |
| ⚙️ | `_parse_html` | `(self, html: str) -> tuple[Optional[str], str, list[str]]` | 570 | Parse HTML content. |
| ⚙️ | `get_sitemaps` | `(self, base_url: str) -> list[str]` | 646 | Get sitemap URLs from robots.txt. |
| ⚙️ | `test_connection` | `(self) -> bool` | 662 | Test connection to a known URL. |
| ⚙️ | `close` | `(self) -> None` | 677 | Close HTTP session and clear caches. |

---

## `argus/knowledge/embeddings.py`

### 🏛️ `class EmbeddingGenerator()`  <sub>line 23</sub>

> Generate embeddings for chunks.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None, batch_size: int = 32, normalize: bool = True)` | 36 | Initialize embedding generator. |
| ⚙️ | `_load_model` | `(self) -> None` | 60 | Lazy load the embedding model. |
| ⚙️ `@property` | `dimension` | `(self) -> int` | 84 | Get embedding dimension. |
| ⚙️ | `embed_texts` | `(self, texts: list[str]) -> np.ndarray` | 89 | Generate embeddings for texts. |
| ⚙️ | `embed_query` | `(self, query: str) -> list[float]` | 111 | Generate embedding for a query. |
| ⚙️ | `embed_chunks` | `(self, chunks: list[Chunk]) -> list[Embedding]` | 131 | Generate embeddings for chunks. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_embeddings` | `(chunks: list[Chunk], model_name: str = 'all-MiniLM-L6-v2') -> list[Embedding]` | 170 | Convenience function to generate embeddings. |
| 🔧 | `cosine_similarity` | `(v1: list[float], v2: list[float]) -> float` | 188 | Compute cosine similarity between two vectors. |
| 🔧 | `batch_cosine_similarity` | `(query: list[float], vectors: list[list[float]]) -> list[float]` | 212 | Compute cosine similarity between query and multiple vectors. |

---

## `argus/knowledge/indexing.py`

### 🏛️ `class SearchResult()`  <sub>line 22</sub>

> Decorators: `@dataclass`  
> Result from index search.  

### 🏛️ `class BM25Index()`  <sub>line 31</sub>

> BM25 sparse text index.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, k1: float = 1.5, b: float = 0.75)` | 43 | Initialize BM25 index. |
| ⚙️ | `_tokenize` | `(self, text: str) -> list[str]` | 63 | Simple whitespace tokenization. |
| ⚙️ | `add_chunks` | `(self, chunks: list[Chunk]) -> None` | 71 | Add chunks to the index. |
| ⚙️ | `search` | `(self, query: str, top_k: int = 10) -> list[SearchResult]` | 95 | Search the index. |
| ⚙️ | `clear` | `(self) -> None` | 132 | Clear the index. |

### 🏛️ `class VectorIndex()`  <sub>line 140</sub>

> FAISS vector index for dense retrieval.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, dimension: int = 384, index_type: str = 'flat')` | 152 | Initialize vector index. |
| ⚙️ | `_init_index` | `(self) -> None` | 173 | Initialize FAISS index. |
| ⚙️ | `add_vectors` | `(self, chunk_ids: list[str], vectors: list[list[float]], chunks: Optional[list[Chunk]] = None) -> None` | 193 | Add vectors to the index. |
| ⚙️ | `search` | `(self, query_vector: list[float], top_k: int = 10) -> list[SearchResult]` | 232 | Search the index. |
| ⚙️ | `clear` | `(self) -> None` | 270 | Clear the index. |

### 🏛️ `class HybridIndex()`  <sub>line 277</sub>

> Hybrid BM25 + Vector index.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, dimension: int = 384, lambda_param: float = 0.7)` | 290 | Initialize hybrid index. |
| ⚙️ | `add_chunks` | `(self, chunks: list[Chunk], vectors: list[list[float]]) -> None` | 310 | Add chunks with their embeddings. |
| ⚙️ | `search` | `(self, query: str, query_vector: list[float], top_k: int = 10, rerank_k: int = 50) -> list[SearchResult]` | 337 | Hybrid search combining BM25 and vector similarity. |
| ⚙️ | `normalize` | `(scores: dict[str, float]) -> dict[str, float]` | 365 |  |
| ⚙️ | `clear` | `(self) -> None` | 407 | Clear all indexes. |

---

## `argus/knowledge/ingestion.py`

### 🏛️ `class DocumentLoader()`  <sub>line 59</sub>

> Universal document loader.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, compute_checksums: bool = True, max_content_length: Optional[int] = None)` | 73 | Initialize document loader. |
| ⚙️ | `load` | `(self, source: Union[str, Path], title: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> Document` | 88 | Load a document from file path or URL. |
| ⚙️ | `_load_pdf` | `(self, path: str, title: Optional[str], metadata: Optional[dict[str, Any]]) -> Document` | 119 | Load PDF document using PyMuPDF. |
| ⚙️ | `_load_html` | `(self, path_or_url: str, title: Optional[str], metadata: Optional[dict[str, Any]]) -> Document` | 174 | Load HTML document using BeautifulSoup. |
| ⚙️ | `_load_csv` | `(self, path: str, title: Optional[str], metadata: Optional[dict[str, Any]]) -> Document` | 241 | Load CSV as text content. |
| ⚙️ | `_load_json` | `(self, path: str, title: Optional[str], metadata: Optional[dict[str, Any]]) -> Document` | 278 | Load JSON as text content. |
| ⚙️ | `_load_text` | `(self, path: str, title: Optional[str], metadata: Optional[dict[str, Any]]) -> Document` | 306 | Load plain text file. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `compute_checksum` | `(content: str) -> str` | 26 | Compute SHA-256 checksum of content. |
| 🔧 | `detect_source_type` | `(path_or_url: str) -> SourceType` | 31 | Detect source type from file path or URL. |
| 🔧 | `load_document` | `(source: Union[str, Path], title: Optional[str] = None, **kwargs: Any) -> Document` | 345 | Load a document from file path or URL. |
| 🔧 | `load_pdf` | `(path: Union[str, Path], title: Optional[str] = None) -> Document` | 365 | Load a PDF document. |
| 🔧 | `load_html` | `(path_or_url: str, title: Optional[str] = None) -> Document` | 374 | Load an HTML document or URL. |
| 🔧 | `load_text` | `(path: Union[str, Path], title: Optional[str] = None) -> Document` | 383 | Load a text file. |

---

## `argus/mcp/client.py`

### 🏛️ `class MCPToolInfo()`  <sub>line 22</sub>

> Decorators: `@dataclass`  
> Information about a remote MCP tool.  

### 🏛️ `class MCPResourceInfo()`  <sub>line 30</sub>

> Decorators: `@dataclass`  
> Information about a remote MCP resource.  

### 🏛️ `class MCPClient()`  <sub>line 38</sub>

> Client for connecting to MCP servers.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[MCPClientConfig] = None)` | 48 |  |
| ⚙️ | `connect` | `(self, *command: str) -> bool` | 57 | Connect to an MCP server via stdio. |
| ⚙️ | `_send_request` | `(self, method: str, params: dict[str, Any]) -> Optional[dict[str, Any]]` | 75 | Send a JSON-RPC request. |
| ⚙️ | `list_tools` | `(self) -> List[MCPToolInfo]` | 92 | List available tools from the server. |
| ⚙️ | `call_tool` | `(self, name: str, **arguments: Any) -> Any` | 105 | Call a tool on the remote server. |
| ⚙️ | `list_resources` | `(self) -> List[MCPResourceInfo]` | 119 | List available resources from the server. |
| ⚙️ | `read_resource` | `(self, uri: str) -> Any` | 133 | Read a resource from the server. |
| ⚙️ | `disconnect` | `(self) -> None` | 146 | Disconnect from the server. |
| ⚙️ `@property` | `is_connected` | `(self) -> bool` | 156 |  |
| ⚙️ | `__enter__` | `(self) -> 'MCPClient'` | 159 |  |
| ⚙️ | `__exit__` | `(self, *args: Any) -> None` | 162 |  |

---

## `argus/mcp/config.py`

### 🏛️ `class TransportType(str, Enum)`  <sub>line 18</sub>

> MCP transport types.  

### 🏛️ `class MCPServerConfig(BaseModel)`  <sub>line 25</sub>

> Configuration for MCP server.  

### 🏛️ `class MCPClientConfig(BaseModel)`  <sub>line 39</sub>

> Configuration for MCP client.  

### 🏛️ `class MCPToolSchema(BaseModel)`  <sub>line 48</sub>

> Schema for an MCP tool.  

### 🏛️ `class MCPResourceSchema(BaseModel)`  <sub>line 56</sub>

> Schema for an MCP resource.  

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_default_server_config` | `() -> MCPServerConfig` | 64 |  |
| 🔧 | `get_default_client_config` | `() -> MCPClientConfig` | 68 |  |

---

## `argus/mcp/resources.py`

### 🏛️ `class BaseResourceAdapter(ABC)`  <sub>line 23</sub>

> Base class for MCP resource adapters.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property`, `@abstractmethod` | `uri` | `(self) -> str` | 28 |  |
| ⚙️ `@property`, `@abstractmethod` | `name` | `(self) -> str` | 33 |  |
| ⚙️ `@property` | `description` | `(self) -> str` | 37 |  |
| ⚙️ `@property` | `mime_type` | `(self) -> str` | 41 |  |
| ⚙️ `@abstractmethod` | `read` | `(self) -> Any` | 45 |  |
| ⚙️ | `get_schema` | `(self) -> MCPResourceSchema` | 48 |  |

### 🏛️ `class CDAGResource(BaseResourceAdapter)`  <sub>line 55</sub>

> Expose CDAG debate graph as MCP resource.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, graph: 'CDAG', resource_id: str = 'default')` | 58 |  |
| ⚙️ `@property` | `uri` | `(self) -> str` | 63 |  |
| ⚙️ `@property` | `name` | `(self) -> str` | 67 |  |
| ⚙️ `@property` | `description` | `(self) -> str` | 71 |  |
| ⚙️ | `read` | `(self) -> Dict[str, Any]` | 74 | Read the CDAG structure. |

### 🏛️ `class EvidenceResource(BaseResourceAdapter)`  <sub>line 95</sub>

> Expose evidence nodes from CDAG as resource.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, graph: 'CDAG', resource_id: str = 'default')` | 98 |  |
| ⚙️ `@property` | `uri` | `(self) -> str` | 103 |  |
| ⚙️ `@property` | `name` | `(self) -> str` | 107 |  |
| ⚙️ `@property` | `description` | `(self) -> str` | 111 |  |
| ⚙️ | `read` | `(self) -> Dict[str, Any]` | 114 |  |

### 🏛️ `class ProvenanceResource(BaseResourceAdapter)`  <sub>line 127</sub>

> Expose provenance ledger as MCP resource.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, ledger: 'ProvenanceLedger')` | 130 |  |
| ⚙️ `@property` | `uri` | `(self) -> str` | 134 |  |
| ⚙️ `@property` | `name` | `(self) -> str` | 138 |  |
| ⚙️ `@property` | `description` | `(self) -> str` | 142 |  |
| ⚙️ | `read` | `(self) -> Dict[str, Any]` | 145 |  |

### 🏛️ `class ConfigResource(BaseResourceAdapter)`  <sub>line 153</sub>

> Expose current ARGUS configuration as resource.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Any = None)` | 156 |  |
| ⚙️ `@property` | `uri` | `(self) -> str` | 160 |  |
| ⚙️ `@property` | `name` | `(self) -> str` | 164 |  |
| ⚙️ `@property` | `description` | `(self) -> str` | 168 |  |
| ⚙️ | `read` | `(self) -> Dict[str, Any]` | 171 |  |

### 🏛️ `class ResourceRegistry()`  <sub>line 182</sub>

> Registry for MCP resources.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 185 |  |
| ⚙️ | `register` | `(self, adapter: BaseResourceAdapter) -> None` | 188 |  |
| ⚙️ | `get` | `(self, uri: str) -> Optional[BaseResourceAdapter]` | 192 |  |
| ⚙️ | `list_all` | `(self) -> list[MCPResourceSchema]` | 195 |  |
| ⚙️ | `read` | `(self, uri: str) -> Optional[Any]` | 198 |  |

---

## `argus/mcp/server.py`

### 🏛️ `class MCPRequest()`  <sub>line 25</sub>

> Decorators: `@dataclass`  
> MCP JSON-RPC request.  

### 🏛️ `class MCPResponse()`  <sub>line 34</sub>

> Decorators: `@dataclass`  
> MCP JSON-RPC response.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 41 |  |

### 🏛️ `class ArgusServer()`  <sub>line 50</sub>

> MCP Server for ARGUS.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[MCPServerConfig] = None, name: str = 'argus-server')` | 69 |  |
| ⚙️ | `tool` | `(self, name: Optional[str] = None, description: str = '') -> Callable` | 78 | Decorator to register a function as an MCP tool. |
| ⚙️ | `decorator` | `(func: Callable) -> Callable` | 80 |  |
| ⚙️ | `resource` | `(self, uri: str, name: Optional[str] = None, description: str = '') -> Callable` | 90 | Decorator to register a function as an MCP resource. |
| ⚙️ | `decorator` | `(func: Callable) -> Callable` | 92 |  |
| ⚙️ | `register_argus_tool` | `(self, tool: 'BaseTool') -> None` | 103 | Register an existing ARGUS tool. |
| ⚙️ | `wrapper` | `(**kwargs: Any) -> Any` | 105 |  |
| ⚙️ | `_generate_tool_schema` | `(self, func: Callable, name: str, description: str) -> MCPToolSchema` | 115 | Generate JSON schema from function signature. |
| ⚙️ | `handle_request` | `(self, request: MCPRequest) -> MCPResponse` | 138 | Handle an MCP request. |
| ⚙️ | `_handle_initialize` | `(self, request: MCPRequest) -> MCPResponse` | 157 |  |
| ⚙️ | `_handle_tools_list` | `(self, request: MCPRequest) -> MCPResponse` | 167 |  |
| ⚙️ | `_handle_tools_call` | `(self, request: MCPRequest) -> MCPResponse` | 172 |  |
| ⚙️ | `_handle_resources_list` | `(self, request: MCPRequest) -> MCPResponse` | 183 |  |
| ⚙️ | `_handle_resources_read` | `(self, request: MCPRequest) -> MCPResponse` | 188 |  |
| ⚙️ | `run` | `(self) -> None` | 198 | Run the MCP server (STDIO transport). |
| ⚙️ | `_run_stdio` | `(self) -> None` | 205 | Run with stdio transport. |
| ⚙️ | `stop` | `(self) -> None` | 221 | Stop the server. |

---

## `argus/mcp/tools.py`

### 🏛️ `class ToolAdapter()`  <sub>line 21</sub>

> Convert ARGUS BaseTool to MCP tool format.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@staticmethod` | `to_mcp_schema` | `(tool: 'BaseTool') -> MCPToolSchema` | 25 | Convert ARGUS tool to MCP schema. |
| ⚙️ `@staticmethod` | `create_mcp_handler` | `(tool: 'BaseTool')` | 35 | Create MCP-compatible handler from ARGUS tool. |
| ⚙️ | `handler` | `(**kwargs: Any) -> Dict[str, Any]` | 37 |  |

### 🏛️ `class ToolSchemaGenerator()`  <sub>line 48</sub>

> Generate JSON schemas for MCP tools.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@staticmethod` | `from_function` | `(func, name: Optional[str] = None, description: Optional[str] = None) -> MCPToolSchema` | 52 | Generate schema from Python function. |

### 🏛️ `class MCPToolWrapper()`  <sub>line 82</sub>

> Wrap an MCP tool as an ARGUS-compatible tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, name: str, description: str, client: Any, input_schema: Optional[dict] = None)` | 85 |  |
| ⚙️ | `execute` | `(self, **kwargs: Any) -> 'ToolResult'` | 91 | Execute the remote MCP tool. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 100 |  |
| ⚙️ | `__call__` | `(self, **kwargs: Any) -> 'ToolResult'` | 103 |  |

### 🏛️ `class ToolRegistry()`  <sub>line 107</sub>

> Registry for managing tool conversions.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 110 |  |
| ⚙️ | `register_argus_tool` | `(self, tool: 'BaseTool') -> MCPToolSchema` | 114 | Register ARGUS tool and return MCP schema. |
| ⚙️ | `register_mcp_tool` | `(self, name: str, description: str, client: Any, schema: dict = None) -> MCPToolWrapper` | 119 | Register external MCP tool as ARGUS tool. |
| ⚙️ | `get_argus_tool` | `(self, name: str) -> Optional['BaseTool']` | 125 |  |
| ⚙️ | `get_mcp_wrapper` | `(self, name: str) -> Optional[MCPToolWrapper]` | 128 |  |

---

## `argus/memory/config.py`

### 🏛️ `class MemoryType(str, Enum)`  <sub>line 19</sub>

> Types of memory.  

### 🏛️ `class StorageBackend(str, Enum)`  <sub>line 30</sub>

> Storage backend options.  

### 🏛️ `class ShortTermConfig(BaseModel)`  <sub>line 38</sub>

> Configuration for short-term memory.  

### 🏛️ `class LongTermConfig(BaseModel)`  <sub>line 47</sub>

> Configuration for long-term memory.  

### 🏛️ `class SemanticCacheConfig(BaseModel)`  <sub>line 57</sub>

> Configuration for semantic cache.  

### 🏛️ `class MemoryConfig(BaseModel)`  <sub>line 66</sub>

> Main memory configuration.  

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_default_memory_config` | `() -> MemoryConfig` | 76 | Get default memory configuration. |

---

## `argus/memory/long_term.py`

### 🏛️ `class MemoryEntry()`  <sub>line 24</sub>

> Decorators: `@dataclass`  
> A long-term memory entry.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 36 |  |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'MemoryEntry'` | 43 |  |

### 🏛️ `class BaseLongTermMemory(ABC)`  <sub>line 50</sub>

> Abstract base for long-term memory.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@abstractmethod` | `add` | `(self, content: str, memory_type: str = 'general', **kwargs: Any) -> str` | 54 |  |
| ⚙️ `@abstractmethod` | `search` | `(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]` | 58 |  |
| ⚙️ `@abstractmethod` | `get` | `(self, memory_id: str) -> Optional[MemoryEntry]` | 62 |  |
| ⚙️ `@abstractmethod` | `delete` | `(self, memory_id: str) -> bool` | 66 |  |

### 🏛️ `class VectorStoreMemory(BaseLongTermMemory)`  <sub>line 70</sub>

> Vector store backed long-term memory using FAISS.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, embedding_generator: Optional[Any] = None, persistence_path: Optional[str] = None, dimension: int = 384)` | 73 |  |
| ⚙️ | `_init_index` | `(self) -> None` | 86 |  |
| ⚙️ | `_get_embedding` | `(self, text: str) -> np.ndarray` | 94 |  |
| ⚙️ | `add` | `(self, content: str, memory_type: str = 'general', importance: float = 0.5, **kwargs: Any) -> str` | 100 |  |
| ⚙️ | `search` | `(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]` | 116 |  |
| ⚙️ | `get` | `(self, memory_id: str) -> Optional[MemoryEntry]` | 146 |  |
| ⚙️ | `delete` | `(self, memory_id: str) -> bool` | 149 |  |
| ⚙️ | `save` | `(self) -> None` | 155 |  |
| ⚙️ | `_load` | `(self) -> None` | 162 |  |

### 🏛️ `class SemanticMemory(VectorStoreMemory)`  <sub>line 180</sub>

> Memory for facts and knowledge.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `add_fact` | `(self, fact: str, source: Optional[str] = None, **kwargs: Any) -> str` | 183 |  |

### 🏛️ `class EpisodicMemory(VectorStoreMemory)`  <sub>line 187</sub>

> Memory for past interactions and events.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `add_episode` | `(self, description: str, participants: Optional[List[str]] = None, **kwargs: Any) -> str` | 190 |  |

### 🏛️ `class ProceduralMemory(VectorStoreMemory)`  <sub>line 194</sub>

> Memory for learned behaviors and patterns.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `add_procedure` | `(self, procedure: str, success_rate: float = 0.0, **kwargs: Any) -> str` | 197 |  |

---

## `argus/memory/semantic_cache.py`

### 🏛️ `class CacheEntry()`  <sub>line 23</sub>

> Decorators: `@dataclass`  
> A semantic cache entry.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `is_expired` | `(self) -> bool` | 35 |  |

### 🏛️ `class SemanticCache()`  <sub>line 41</sub>

> Cache that retrieves based on semantic similarity.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, similarity_threshold: float = 0.85, max_entries: int = 1000, ttl_seconds: int = 3600, embedding_generator: Optional[Any] = None, dimension: int = 384)` | 44 |  |
| ⚙️ | `_get_embedding` | `(self, text: str) -> np.ndarray` | 58 |  |
| ⚙️ | `_generate_id` | `(self, query: str) -> str` | 64 |  |
| ⚙️ | `get` | `(self, query: str) -> Optional[str]` | 67 | Get cached response for query if similar enough. |
| ⚙️ | `set` | `(self, query: str, response: str, **metadata: Any) -> str` | 92 | Cache a response for a query. |
| ⚙️ | `invalidate` | `(self, cache_id: str) -> bool` | 111 | Remove entry from cache. |
| ⚙️ | `_cleanup_expired` | `(self) -> int` | 121 | Remove expired entries. |
| ⚙️ | `_evict_lru` | `(self) -> None` | 128 | Evict least recently used entry. |
| ⚙️ | `clear` | `(self) -> None` | 135 | Clear all cache entries. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 141 | Get cache statistics. |

---

## `argus/memory/short_term.py`

### 🏛️ `class Message()`  <sub>line 19</sub>

> Decorators: `@dataclass`  
> A conversation message.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 26 |  |

### 🏛️ `class BaseShortTermMemory(ABC)`  <sub>line 31</sub>

> Abstract base for short-term memory.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_messages: int = 100)` | 34 |  |
| ⚙️ `@abstractmethod` | `add` | `(self, role: str, content: str, **kwargs: Any) -> None` | 39 |  |
| ⚙️ `@abstractmethod` | `get_messages` | `(self) -> List[Message]` | 43 |  |
| ⚙️ | `clear` | `(self) -> None` | 46 |  |
| ⚙️ | `__len__` | `(self) -> int` | 49 |  |

### 🏛️ `class ConversationBufferMemory(BaseShortTermMemory)`  <sub>line 53</sub>

> Full conversation history buffer.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `add` | `(self, role: str, content: str, **kwargs: Any) -> None` | 56 |  |
| ⚙️ | `get_messages` | `(self) -> List[Message]` | 62 |  |
| ⚙️ | `get_context_string` | `(self) -> str` | 65 | Get messages as formatted string. |

### 🏛️ `class ConversationWindowMemory(BaseShortTermMemory)`  <sub>line 70</sub>

> Sliding window memory - keeps last K messages.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, window_size: int = 10, max_messages: int = 100)` | 73 |  |
| ⚙️ | `add` | `(self, role: str, content: str, **kwargs: Any) -> None` | 77 |  |
| ⚙️ | `get_messages` | `(self) -> List[Message]` | 83 |  |
| ⚙️ | `get_full_history` | `(self) -> List[Message]` | 86 |  |

### 🏛️ `class ConversationSummaryMemory(BaseShortTermMemory)`  <sub>line 90</sub>

> Memory that maintains a running summary.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_messages: int = 100, summary_threshold: int = 10, summarizer: Optional[Any] = None)` | 93 |  |
| ⚙️ | `add` | `(self, role: str, content: str, **kwargs: Any) -> None` | 101 |  |
| ⚙️ | `_update_summary` | `(self) -> None` | 108 |  |
| ⚙️ | `get_messages` | `(self) -> List[Message]` | 123 |  |
| ⚙️ | `get_summary` | `(self) -> str` | 126 |  |
| ⚙️ | `get_context` | `(self) -> str` | 129 |  |

### 🏛️ `class EntityMemory(BaseShortTermMemory)`  <sub>line 136</sub>

> Memory that tracks entities mentioned in conversation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_messages: int = 100)` | 139 |  |
| ⚙️ | `add` | `(self, role: str, content: str, entities: Optional[dict[str, Any]] = None, **kwargs: Any) -> None` | 143 |  |
| ⚙️ | `get_messages` | `(self) -> List[Message]` | 153 |  |
| ⚙️ | `get_entity` | `(self, name: str) -> Optional[dict[str, Any]]` | 156 |  |
| ⚙️ | `get_all_entities` | `(self) -> dict[str, dict[str, Any]]` | 159 |  |
| ⚙️ | `update_entity` | `(self, name: str, info: dict[str, Any]) -> None` | 162 |  |

### 🏛️ `class ShortTermMemoryManager()`  <sub>line 169</sub>

> Manager for short-term memory with multiple strategies.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, memory_type: str = 'buffer', **kwargs: Any)` | 172 |  |
| ⚙️ | `add` | `(self, role: str, content: str, **kwargs: Any) -> None` | 185 |  |
| ⚙️ | `get_messages` | `(self) -> List[Message]` | 188 |  |
| ⚙️ | `clear` | `(self) -> None` | 191 |  |

---

## `argus/memory/store.py`

### 🏛️ `class MemoryStore(ABC)`  <sub>line 21</sub>

> Abstract base class for memory storage backends.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@abstractmethod` | `save` | `(self, key: str, data: dict[str, Any]) -> bool` | 25 |  |
| ⚙️ `@abstractmethod` | `load` | `(self, key: str) -> Optional[dict[str, Any]]` | 29 |  |
| ⚙️ `@abstractmethod` | `delete` | `(self, key: str) -> bool` | 33 |  |
| ⚙️ `@abstractmethod` | `list_keys` | `(self) -> List[str]` | 37 |  |
| ⚙️ `@abstractmethod` | `clear` | `(self) -> int` | 41 |  |

### 🏛️ `class InMemoryStore(MemoryStore)`  <sub>line 45</sub>

> Simple in-memory storage for testing/development.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 48 |  |
| ⚙️ | `save` | `(self, key: str, data: dict[str, Any]) -> bool` | 51 |  |
| ⚙️ | `load` | `(self, key: str) -> Optional[dict[str, Any]]` | 55 |  |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 58 |  |
| ⚙️ | `list_keys` | `(self) -> List[str]` | 64 |  |
| ⚙️ | `clear` | `(self) -> int` | 67 |  |

### 🏛️ `class SQLiteStore(MemoryStore)`  <sub>line 73</sub>

> SQLite-based persistent storage.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, db_path: str = ':memory:')` | 76 |  |
| ⚙️ | `_init_db` | `(self) -> None` | 81 |  |
| ⚙️ | `save` | `(self, key: str, data: dict[str, Any]) -> bool` | 92 |  |
| ⚙️ | `load` | `(self, key: str) -> Optional[dict[str, Any]]` | 105 |  |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 112 |  |
| ⚙️ | `list_keys` | `(self) -> List[str]` | 117 |  |
| ⚙️ | `clear` | `(self) -> int` | 121 |  |
| ⚙️ | `close` | `(self) -> None` | 126 |  |

### 🏛️ `class FAISSStore(MemoryStore)`  <sub>line 130</sub>

> FAISS vector store for embeddings.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, dimension: int = 384, persistence_path: Optional[str] = None)` | 133 |  |
| ⚙️ | `_init_index` | `(self) -> None` | 142 |  |
| ⚙️ | `save` | `(self, key: str, data: dict[str, Any]) -> bool` | 149 |  |
| ⚙️ | `load` | `(self, key: str) -> Optional[dict[str, Any]]` | 160 |  |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 168 |  |
| ⚙️ | `list_keys` | `(self) -> List[str]` | 178 |  |
| ⚙️ | `clear` | `(self) -> int` | 181 |  |
| ⚙️ | `search` | `(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple[str, float]]` | 189 |  |

### 🏛️ `class FileSystemStore(MemoryStore)`  <sub>line 201</sub>

> File-based JSON storage.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, base_path: str)` | 204 |  |
| ⚙️ | `_get_path` | `(self, key: str) -> Path` | 208 |  |
| ⚙️ | `save` | `(self, key: str, data: dict[str, Any]) -> bool` | 212 |  |
| ⚙️ | `load` | `(self, key: str) -> Optional[dict[str, Any]]` | 221 |  |
| ⚙️ | `delete` | `(self, key: str) -> bool` | 227 |  |
| ⚙️ | `list_keys` | `(self) -> List[str]` | 234 |  |
| ⚙️ | `clear` | `(self) -> int` | 237 |  |

---

## `argus/metrics/collector.py`

### 🏛️ `class MetricType(str, Enum)`  <sub>line 35</sub>

> Types of metrics.  

### 🏛️ `class Metric()`  <sub>line 43</sub>

> Decorators: `@dataclass`  
> A recorded metric value.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 51 |  |

### 🏛️ `class MetricsCollector()`  <sub>line 61</sub>

> Collects and aggregates metrics.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_history: int = 10000)` | 79 | Initialize collector. |
| ⚙️ | `set_default_tags` | `(self, tags: dict[str, str]) -> None` | 99 | Set default tags for all metrics. |
| ⚙️ | `record` | `(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, tags: Optional[dict[str, str]] = None) -> None` | 108 | Record a metric value. |
| ⚙️ | `record_counter` | `(self, name: str, value: float = 1, tags: Optional[dict[str, str]] = None) -> None` | 151 | Record a counter increment. |
| ⚙️ | `record_gauge` | `(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None` | 166 | Record a gauge value. |
| ⚙️ | `record_histogram` | `(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None` | 181 | Record a histogram observation. |
| ⚙️ | `get_counter` | `(self, name: str) -> float` | 196 | Get current counter value. |
| ⚙️ | `get_gauge` | `(self, name: str) -> Optional[float]` | 208 | Get current gauge value. |
| ⚙️ | `get_histogram_stats` | `(self, name: str) -> dict[str, float]` | 220 | Get histogram statistics. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 249 | Get all metrics statistics. |
| ⚙️ | `get_history` | `(self, name: Optional[str] = None, metric_type: Optional[MetricType] = None, since: Optional[datetime] = None, limit: int = 100) -> list[Metric]` | 266 | Get metric history. |
| ⚙️ | `reset` | `(self) -> None` | 296 | Reset all metrics. |
| ⚙️ | `export` | `(self) -> list[dict[str, Any]]` | 305 | Export all metrics as dicts. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_default_collector` | `() -> MetricsCollector` | 323 | Get the default metrics collector. |
| 🔧 | `record_metric` | `(name: str, value: float, metric_type: MetricType = MetricType.GAUGE, tags: Optional[dict[str, str]] = None) -> None` | 337 | Record a metric in the default collector. |
| 🔧 | `record_counter` | `(name: str, value: float = 1, tags: Optional[dict[str, str]] = None) -> None` | 347 | Record a counter in the default collector. |
| 🔧 | `record_gauge` | `(name: str, value: float, tags: Optional[dict[str, str]] = None) -> None` | 356 | Record a gauge in the default collector. |
| 🔧 | `record_histogram` | `(name: str, value: float, tags: Optional[dict[str, str]] = None) -> None` | 365 | Record a histogram in the default collector. |

---

## `argus/metrics/traces.py`

### 🏛️ `class SpanStatus(str, Enum)`  <sub>line 41</sub>

> Status of a span.  

### 🏛️ `class TraceConfig(BaseModel)`  <sub>line 48</sub>

> Configuration for tracing.  

### 🏛️ `class SpanContext()`  <sub>line 78</sub>

> Decorators: `@dataclass`  
> Context for a trace span.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 84 |  |

### 🏛️ `class Span()`  <sub>line 93</sub>

> Decorators: `@dataclass`  
> A trace span representing a unit of work.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `set_attribute` | `(self, key: str, value: Any) -> 'Span'` | 107 | Set an attribute on the span. |
| ⚙️ | `set_attributes` | `(self, attributes: dict[str, Any]) -> 'Span'` | 120 | Set multiple attributes. |
| ⚙️ | `add_event` | `(self, name: str, attributes: Optional[dict[str, Any]] = None) -> 'Span'` | 132 | Add an event to the span. |
| ⚙️ | `set_status` | `(self, status: SpanStatus, message: Optional[str] = None) -> 'Span'` | 153 | Set the span status. |
| ⚙️ | `end` | `(self) -> None` | 168 | End the span. |
| ⚙️ `@property` | `duration_ms` | `(self) -> float` | 175 | Get span duration in milliseconds. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 183 | Convert span to dictionary. |

### 🏛️ `class Tracer()`  <sub>line 197</sub>

> Creates and manages trace spans.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, service_name: str = 'argus', config: Optional[TraceConfig] = None)` | 214 | Initialize tracer. |
| ⚙️ | `_generate_id` | `(self) -> str` | 233 | Generate a unique ID. |
| ⚙️ | `_should_sample` | `(self) -> bool` | 237 | Check if trace should be sampled. |
| ⚙️ `@contextmanager` | `span` | `(self, name: str, attributes: Optional[dict[str, Any]] = None)` | 247 | Create a new span as a context manager. |
| ⚙️ | `start_span` | `(self, name: str, attributes: Optional[dict[str, Any]] = None) -> Span` | 317 | Start a span manually (must call end()). |
| ⚙️ | `get_current_span` | `(self) -> Optional[Span]` | 351 | Get the current active span. |
| ⚙️ | `get_trace` | `(self, trace_id: str) -> list[Span]` | 359 | Get all spans for a trace. |
| ⚙️ | `get_recent_spans` | `(self, limit: int = 100) -> list[Span]` | 371 | Get recent spans. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 383 | Get tracer statistics. |
| ⚙️ | `export` | `(self) -> list[dict[str, Any]]` | 400 | Export all spans as dicts. |
| ⚙️ | `clear` | `(self) -> None` | 409 | Clear all stored spans. |

### 🏛️ `class _NoOpSpan()`  <sub>line 416</sub>

> No-op span for when sampling is disabled.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `set_attribute` | `(self, key: str, value: Any) -> '_NoOpSpan'` | 419 |  |
| ⚙️ | `set_attributes` | `(self, attributes: dict[str, Any]) -> '_NoOpSpan'` | 422 |  |
| ⚙️ | `add_event` | `(self, name: str, attributes: Optional[dict[str, Any]] = None) -> '_NoOpSpan'` | 425 |  |
| ⚙️ | `set_status` | `(self, status: SpanStatus, message: Optional[str] = None) -> '_NoOpSpan'` | 428 |  |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_tracer` | `(service_name: str = 'argus') -> Tracer` | 440 | Get the default tracer. |

---

## `argus/orchestrator.py`

### 🏛️ `class DebateResult()`  <sub>line 36</sub>

> Decorators: `@dataclass`  
> Result of a complete debate.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 57 | Convert to dictionary. |

### 🏛️ `class RDCOrchestrator()`  <sub>line 69</sub>

> Research Debate Chain orchestrator.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, llm: Optional['BaseLLM'] = None, ledger: Optional[ProvenanceLedger] = None, max_rounds: int = 5)` | 85 | Initialize orchestrator. |
| ⚙️ | `debate` | `(self, proposition_text: str, prior: float = 0.5, retriever: Optional['HybridRetriever'] = None, domain: str = 'general') -> DebateResult` | 113 | Run a complete debate on a proposition. |
| ⚙️ | `quick_evaluate` | `(self, proposition_text: str, prior: float = 0.5) -> Verdict` | 231 | Quick evaluation without full debate. |
| ⚙️ `@property` | `graph` | `(self) -> Optional[CDAG]` | 286 | Get current debate graph. |
| ⚙️ `@property` | `agenda` | `(self) -> Optional[DebateAgenda]` | 291 | Get current agenda. |

---

## `argus/outputs/plotting.py`

### 🏛️ `class PlotTheme(str, Enum)`  <sub>line 60</sub>

> Available plot themes.  

### 🏛️ `class PlotConfig()`  <sub>line 122</sub>

> Decorators: `@dataclass`  
> Configuration for plot generation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 135 |  |

### 🏛️ `class DebatePlotter()`  <sub>line 265</sub>

> Main plotting interface for ARGUS debate results.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[PlotConfig] = None)` | 279 | Initialize the plotter with configuration. |
| ⚙️ | `_save_figure` | `(self, fig: plt.Figure, name: str) -> Path` | 285 | Save figure with configured settings. |
| ⚙️ | `plot_posterior_evolution` | `(self, result: dict, title: Optional[str] = None) -> Path` | 298 | Create posterior probability evolution plot across debate rounds. |
| ⚙️ | `plot_evidence_distribution` | `(self, result: dict, title: Optional[str] = None) -> Path` | 372 | Create evidence polarity distribution donut chart. |
| ⚙️ | `plot_specialist_contributions` | `(self, result: dict, title: Optional[str] = None) -> Path` | 449 | Create stacked bar chart of specialist contributions per round. |
| ⚙️ | `plot_confidence_distribution` | `(self, result: dict, title: Optional[str] = None) -> Path` | 515 | Create histogram and KDE plot of evidence confidence scores. |
| ⚙️ | `plot_round_heatmap` | `(self, result: dict, title: Optional[str] = None) -> Path` | 584 | Create heatmap of evidence by specialist and round. |
| ⚙️ | `plot_cdag_network` | `(self, result: dict, title: Optional[str] = None) -> Path` | 647 | Create CDAG network visualization using NetworkX. |
| ⚙️ | `plot_multi_stock_comparison` | `(self, results: list[dict], title: Optional[str] = None) -> Path` | 760 | Create multi-stock comparison dashboard. |
| ⚙️ | `plot_summary_radar` | `(self, results: list[dict], title: Optional[str] = None) -> Path` | 849 | Create radar chart comparing multiple stocks across metrics. |
| ⚙️ | `generate_all_plots` | `(self, result: dict) -> list[Path]` | 903 | Generate all available plots for a single debate result. |
| ⚙️ | `generate_all_comparison_plots` | `(self, results: list[dict]) -> list[Path]` | 935 | Generate all comparison plots for multiple debate results. |

### 🏛️ `class InteractivePlotter()`  <sub>line 969</sub>

> Interactive plotting using Plotly for web-based visualizations.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[PlotConfig] = None)` | 974 |  |
| ⚙️ | `_save_html` | `(self, fig: go.Figure, name: str) -> Path` | 980 | Save figure as HTML. |
| ⚙️ | `plot_interactive_posterior` | `(self, result: dict) -> Path` | 987 | Create interactive posterior evolution plot. |
| ⚙️ | `plot_interactive_network` | `(self, result: dict) -> Path` | 1027 | Create interactive network graph. |
| ⚙️ | `plot_dashboard` | `(self, results: list[dict]) -> Path` | 1112 | Create comprehensive interactive dashboard. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `setup_plot_style` | `(theme: PlotTheme = PlotTheme.PUBLICATION)` | 140 | Apply publication-quality matplotlib style. |
| 🔧 | `extract_posterior_timeline` | `(result: dict) -> tuple[list[int], list[float], list[float]]` | 197 | Extract posterior values across rounds. |
| 🔧 | `extract_evidence_by_polarity` | `(result: dict) -> tuple[int, int]` | 211 | Count supporting vs attacking evidence. |
| 🔧 | `extract_specialist_contributions` | `(result: dict) -> dict[str, dict[int, int]]` | 226 | Extract evidence count per specialist per round. |
| 🔧 | `extract_evidence_confidence` | `(result: dict) -> list[float]` | 243 | Extract confidence scores from graph nodes. |
| 🔧 | `extract_graph_structure` | `(result: dict) -> tuple[list[dict], list[dict]]` | 255 | Extract nodes and edges from graph data. |
| 🔧 | `generate_debate_plots` | `(result: dict, output_dir: Optional[Path] = None, interactive: bool = True) -> list[Path]` | 1189 | Generate all plots for a single debate result. |
| 🔧 | `generate_comparison_plots` | `(results: list[dict], output_dir: Optional[Path] = None, interactive: bool = True) -> list[Path]` | 1218 | Generate comparison plots for multiple debate results. |

---

## `argus/outputs/reports.py`

### 🏛️ `class ReportFormat(str, Enum)`  <sub>line 41</sub>

> Available report formats.  

### 🏛️ `class ReportConfig(BaseModel)`  <sub>line 49</sub>

> Configuration for report generation.  

### 🏛️ `class EvidenceSummary()`  <sub>line 86</sub>

> Decorators: `@dataclass`  
> Summary of a piece of evidence.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 95 |  |

### 🏛️ `class VerdictSummary()`  <sub>line 100</sub>

> Decorators: `@dataclass`  
> Summary of a debate verdict.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 108 |  |

### 🏛️ `class DebateReport()`  <sub>line 113</sub>

> Decorators: `@dataclass`  
> Complete report of a debate session.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 145 | Convert report to dictionary. |
| ⚙️ | `to_json` | `(self, indent: Optional[int] = 2) -> str` | 176 | Convert report to JSON string. |
| ⚙️ | `to_markdown` | `(self) -> str` | 187 | Convert report to Markdown format. |
| ⚙️ | `to_summary` | `(self) -> dict[str, Any]` | 257 | Get a compact summary of the report. |

### 🏛️ `class ReportGenerator()`  <sub>line 274</sub>

> Generates reports from debate results.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ReportConfig] = None)` | 290 | Initialize the generator. |
| ⚙️ | `generate` | `(self, result: 'DebateResult', graph: Optional['CDAG'] = None) -> DebateReport` | 299 | Generate a report from a debate result. |
| ⚙️ | `_extract_graph_data` | `(self, graph: 'CDAG') -> dict[str, Any]` | 386 | Extract graph structure for report. |
| ⚙️ | `generate_batch` | `(self, results: list['DebateResult']) -> list[DebateReport]` | 432 | Generate reports for multiple debate results. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `generate_report` | `(result: 'DebateResult', config: Optional[ReportConfig] = None) -> DebateReport` | 451 | Generate a report from a debate result. |
| 🔧 | `export_json` | `(result: 'DebateResult', path: Optional[str] = None, config: Optional[ReportConfig] = None) -> str` | 468 | Export debate result as JSON. |
| 🔧 | `export_markdown` | `(result: 'DebateResult', path: Optional[str] = None, config: Optional[ReportConfig] = None) -> str` | 494 | Export debate result as Markdown. |

---

## `argus/provenance/integrity.py`

### 🏛️ `class Attestation()`  <sub>line 84</sub>

> Decorators: `@dataclass`  
> Attestation for a piece of content.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__post_init__` | `(self)` | 103 |  |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 107 | Convert to dictionary. |
| ⚙️ | `verify` | `(self, content: str) -> bool` | 118 | Verify attestation against content. |

### 🏛️ `class IntegrityChecker()`  <sub>line 193</sub>

> Utility for checking content integrity.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 200 |  |
| ⚙️ | `register` | `(self, content_id: str, content: str, attester: str) -> Attestation` | 203 | Register content and create attestation. |
| ⚙️ | `verify` | `(self, content_id: str, content: str) -> tuple[bool, Optional[str]]` | 224 | Verify content against registered attestation. |
| ⚙️ | `get_attestation` | `(self, content_id: str) -> Optional[Attestation]` | 248 | Get attestation by content ID. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `compute_hash` | `(content: str, algorithm: str = 'sha256') -> str` | 17 | Compute cryptographic hash of content. |
| 🔧 | `verify_hash` | `(content: str, expected_hash: str, algorithm: str = 'sha256') -> bool` | 38 | Verify content matches expected hash. |
| 🔧 | `compute_merkle_root` | `(hashes: list[str]) -> str` | 54 | Compute Merkle root from list of hashes. |
| 🔧 | `create_attestation` | `(content: str, attester: str, algorithm: str = 'sha256', metadata: Optional[dict[str, Any]] = None) -> Attestation` | 131 | Create an attestation for content. |
| 🔧 | `create_batch_attestation` | `(contents: list[str], attester: str, algorithm: str = 'sha256') -> tuple[Attestation, list[str]]` | 160 | Create attestation for multiple contents using Merkle tree. |

---

## `argus/provenance/ledger.py`

### 🏛️ `class EventType(str, Enum)`  <sub>line 24</sub>

> Types of provenance events.  

### 🏛️ `class ProvenanceEvent()`  <sub>line 55</sub>

> Decorators: `@dataclass`  
> A provenance event in the ledger.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 82 | Convert to dictionary. |
| ⚙️ | `to_prov` | `(self) -> dict[str, Any]` | 96 | Convert to PROV-O compatible format. |
| ⚙️ `@classmethod` | `from_dict` | `(cls, data: dict[str, Any]) -> 'ProvenanceEvent'` | 113 | Create from dictionary. |

### 🏛️ `class ProvenanceLedger()`  <sub>line 128</sub>

> Append-only provenance ledger.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, path: Optional[str] = None, enabled: bool = True)` | 145 | Initialize ledger. |
| ⚙️ | `_generate_id` | `(self) -> str` | 168 | Generate unique event ID. |
| ⚙️ | `_compute_hash` | `(self, event: ProvenanceEvent) -> str` | 174 | Compute SHA-256 hash for event. |
| ⚙️ | `record` | `(self, event_type: EventType, agent_id: Optional[str] = None, entity_id: Optional[str] = None, activity_id: Optional[str] = None, attributes: Optional[dict[str, Any]] = None) -> Optional[ProvenanceEvent]` | 189 | Record a provenance event. |
| ⚙️ | `query` | `(self, event_type: Optional[EventType] = None, agent_id: Optional[str] = None, entity_id: Optional[str] = None, since: Optional[datetime] = None, until: Optional[datetime] = None, limit: int = 100) -> list[ProvenanceEvent]` | 245 | Query events from the ledger. |
| ⚙️ | `get_entity_history` | `(self, entity_id: str) -> list[ProvenanceEvent]` | 291 | Get complete history for an entity. |
| ⚙️ | `verify_integrity` | `(self) -> tuple[bool, list[str]]` | 306 | Verify hash chain integrity. |
| ⚙️ | `_append_to_file` | `(self, event: ProvenanceEvent) -> None` | 328 | Append event to file. |
| ⚙️ | `_load` | `(self) -> None` | 333 | Load events from file. |
| ⚙️ | `export_prov` | `(self) -> dict[str, Any]` | 345 | Export ledger in PROV-O JSON format. |
| ⚙️ | `__len__` | `(self) -> int` | 360 |  |

---

## `argus/retrieval/cite_critique.py`

### 🏛️ `class CiteResult()`  <sub>line 22</sub>

> Decorators: `@dataclass`  
> Result from cite & critique prompting.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 43 | Convert to dictionary. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `cite_and_critique` | `(llm: 'BaseLLM', query: str, source_text: str, source_id: str = '') -> CiteResult` | 56 | Extract claims with citations and critique. |
| 🔧 | `extract_claims` | `(llm: 'BaseLLM', proposition: str, source_texts: list[tuple[str, str]], max_claims: int = 5) -> list[CiteResult]` | 133 | Extract multiple claims from multiple sources. |
| 🔧 | `build_evidence_summary` | `(claims: list[CiteResult], max_length: int = 500) -> str` | 169 | Build a summary of evidence claims. |

---

## `argus/retrieval/hybrid.py`

### 🏛️ `class RetrievalResult()`  <sub>line 25</sub>

> Decorators: `@dataclass`  
> Result from hybrid retrieval.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 46 | Convert to dictionary. |

### 🏛️ `class HybridRetriever()`  <sub>line 59</sub>

> Hybrid retrieval engine combining BM25 and vector search.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, embedding_model: str = 'all-MiniLM-L6-v2', lambda_param: float = 0.7, use_reranker: bool = False, reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2')` | 77 | Initialize hybrid retriever. |
| ⚙️ | `_init_components` | `(self) -> None` | 102 | Lazy initialize components. |
| ⚙️ | `_init_reranker` | `(self) -> None` | 111 | Lazy initialize reranker. |
| ⚙️ | `index_chunks` | `(self, chunks: list[Chunk]) -> int` | 122 | Index chunks for retrieval. |
| ⚙️ | `retrieve` | `(self, query: str, top_k: int = 10, rerank_k: int = 50) -> list[RetrievalResult]` | 147 | Retrieve relevant chunks for a query. |
| ⚙️ | `_rerank` | `(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]` | 195 | Rerank results using cross-encoder. |
| ⚙️ | `retrieve_with_rrf` | `(self, query: str, top_k: int = 10, k_constant: int = 60) -> list[RetrievalResult]` | 225 | Retrieve using Reciprocal Rank Fusion. |
| ⚙️ | `clear` | `(self) -> None` | 288 | Clear the index. |
| ⚙️ `@property` | `num_indexed` | `(self) -> int` | 294 | Get number of indexed chunks. |

---

## `argus/retrieval/reranker.py`

### 🏛️ `class RerankResult()`  <sub>line 17</sub>

> Decorators: `@dataclass`  
> Result after reranking.  

### 🏛️ `class CrossEncoderReranker()`  <sub>line 26</sub>

> Cross-encoder reranker for neural re-ranking.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: Optional[str] = None, batch_size: int = 32)` | 38 | Initialize reranker. |
| ⚙️ | `_load_model` | `(self) -> None` | 58 | Lazy load the model. |
| ⚙️ | `score_pairs` | `(self, query: str, documents: list[str]) -> list[float]` | 76 | Score query-document pairs. |
| ⚙️ | `rerank` | `(self, query: str, documents: list[tuple[str, str, float]], top_k: Optional[int] = None) -> list[RerankResult]` | 98 | Rerank documents for a query. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `rerank_results` | `(query: str, documents: list[tuple[str, str, float]], model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', top_k: Optional[int] = None) -> list[RerankResult]` | 145 | Convenience function for reranking. |

---

## `argus/tools/base.py`

### 🏛️ `class ToolCategory(str, Enum)`  <sub>line 40</sub>

> Categories of tools for organization and filtering.  

### 🏛️ `class ToolConfig(BaseModel)`  <sub>line 68</sub>

> Configuration for a tool.  

### 🏛️ `class ToolResult()`  <sub>line 100</sub>

> Decorators: `@dataclass`  
> Result from a tool execution.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 119 | Convert to dictionary for serialization. |
| ⚙️ `@classmethod` | `from_error` | `(cls, error: str) -> 'ToolResult'` | 132 | Create a failed result from an error message. |
| ⚙️ `@classmethod` | `from_data` | `(cls, data: dict[str, Any]) -> 'ToolResult'` | 137 | Create a successful result from data. |

### 🏛️ `class BaseTool(ABC)`  <sub>line 142</sub>

> Abstract base class for all ARGUS tools.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ToolConfig] = None)` | 171 | Initialize the tool. |
| ⚙️ `@abstractmethod` | `execute` | `(self, **kwargs: Any) -> ToolResult` | 183 | Execute the tool with the given arguments. |
| ⚙️ | `validate_input` | `(self, **kwargs: Any) -> Optional[str]` | 196 | Validate input arguments before execution. |
| ⚙️ | `__call__` | `(self, **kwargs: Any) -> ToolResult` | 209 | Execute the tool (callable interface). |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 247 | Get JSON schema describing the tool's parameters. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 263 | Get execution statistics for this tool. |
| ⚙️ | `__repr__` | `(self) -> str` | 281 |  |

### 🏛️ `class EchoTool(BaseTool)`  <sub>line 289</sub>

> Simple echo tool for testing.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `execute` | `(self, message: str = '', **kwargs: Any) -> ToolResult` | 298 | Echo the message back. |
| ⚙️ | `validate_input` | `(self, **kwargs: Any) -> Optional[str]` | 312 |  |

### 🏛️ `class CalculatorTool(BaseTool)`  <sub>line 318</sub>

> Safe calculator tool for mathematical expressions.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `execute` | `(self, expression: str = '', **kwargs: Any) -> ToolResult` | 337 | Evaluate a mathematical expression safely. |
| ⚙️ | `validate_input` | `(self, **kwargs: Any) -> Optional[str]` | 398 |  |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `create_tool` | `(name: str, description: str, execute_fn: Callable[..., ToolResult], category: ToolCategory = ToolCategory.CUSTOM, config: Optional[ToolConfig] = None) -> BaseTool` | 405 | Factory function to create a tool from a function. |

---

## `argus/tools/cache.py`

### 🏛️ `class CacheConfig(BaseModel)`  <sub>line 39</sub>

> Configuration for result cache.  

### 🏛️ `class CacheEntry()`  <sub>line 65</sub>

> Decorators: `@dataclass`  
> A cached result entry.  

### 🏛️ `class ResultCache()`  <sub>line 73</sub>

> LRU cache for tool execution results.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[CacheConfig] = None)` | 89 | Initialize the cache. |
| ⚙️ | `_compute_key` | `(self, tool_name: str, arguments: dict[str, Any]) -> str` | 106 | Compute cache key from tool name and arguments. |
| ⚙️ | `get` | `(self, tool_name: str, arguments: dict[str, Any]) -> Optional[ToolResult]` | 123 | Get a cached result. |
| ⚙️ | `set` | `(self, tool_name: str, arguments: dict[str, Any], result: ToolResult, ttl: Optional[int] = None) -> None` | 162 | Store a result in cache. |
| ⚙️ | `invalidate` | `(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> int` | 195 | Invalidate cached entries. |
| ⚙️ | `clear` | `(self) -> None` | 234 | Clear all cached entries. |
| ⚙️ | `_maybe_cleanup` | `(self) -> None` | 240 | Run cleanup if interval has passed. |
| ⚙️ | `_cleanup_expired` | `(self) -> int` | 249 | Remove expired entries. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 269 | Get cache statistics. |
| ⚙️ | `__len__` | `(self) -> int` | 287 | Get number of cached entries. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `cached_tool` | `(ttl: int = 300, cache: Optional[ResultCache] = None) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]` | 297 | Decorator to add caching to a tool's execute method. |

---

## `argus/tools/executor.py`

### 🏛️ `class ExecutorConfig(BaseModel)`  <sub>line 39</sub>

> Configuration for tool executor.  

### 🏛️ `class ExecutionRecord()`  <sub>line 70</sub>

> Decorators: `@dataclass`  
> Record of a tool execution for auditing.  

### 🏛️ `class ToolExecutor()`  <sub>line 81</sub>

> Executes tools with caching, guardrails, and error handling.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, registry: 'ToolRegistry', config: Optional[ExecutorConfig] = None, cache: Optional['ResultCache'] = None, guardrails: Optional[list['Guardrail']] = None)` | 106 | Initialize the executor. |
| ⚙️ | `run` | `(self, tool_name: str, timeout: Optional[float] = None, skip_cache: bool = False, skip_guardrails: bool = False, **kwargs: Any) -> ToolResult` | 130 | Execute a tool by name. |
| ⚙️ | `_execute_with_timeout` | `(self, tool: Any, kwargs: dict[str, Any], timeout: float) -> ToolResult` | 213 | Execute tool with timeout using thread pool. |
| ⚙️ | `run_batch` | `(self, calls: list[tuple[str, dict[str, Any]]], timeout: Optional[float] = None) -> list[ToolResult]` | 238 | Execute multiple tools in parallel. |
| ⚙️ | `_record_execution` | `(self, tool_name: str, arguments: dict[str, Any], result: ToolResult, started_at: datetime, completed_at: datetime, cached: bool = False, guardrail_blocked: bool = False) -> None` | 261 | Record an execution in history. |
| ⚙️ | `get_history` | `(self, tool_name: Optional[str] = None, limit: int = 100) -> list[ExecutionRecord]` | 288 | Get execution history. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 307 | Get executor statistics. |
| ⚙️ | `add_guardrail` | `(self, guardrail: 'Guardrail') -> None` | 329 | Add a guardrail to the executor. |
| ⚙️ | `set_cache` | `(self, cache: 'ResultCache') -> None` | 338 | Set the result cache. |

---

## `argus/tools/guardrails.py`

### 🏛️ `class GuardrailConfig(BaseModel)`  <sub>line 33</sub>

> Configuration for guardrails.  

### 🏛️ `class GuardrailResult()`  <sub>line 56</sub>

> Decorators: `@dataclass`  
> Result from a guardrail check.  

### 🏛️ `class Guardrail(ABC)`  <sub>line 69</sub>

> Abstract base class for guardrails.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[GuardrailConfig] = None)` | 87 | Initialize guardrail. |
| ⚙️ `@abstractmethod` | `check` | `(self, tool_name: str, arguments: dict[str, Any]) -> GuardrailResult` | 96 | Check if tool execution should be allowed. |
| ⚙️ | `check_output` | `(self, tool_name: str, result: Any) -> GuardrailResult` | 112 | Check tool output (optional). |

### 🏛️ `class ContentFilter(Guardrail)`  <sub>line 135</sub>

> Filters content based on keyword/pattern matching.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, blocked_patterns: Optional[list[str]] = None, blocked_keywords: Optional[list[str]] = None, case_sensitive: bool = False, config: Optional[GuardrailConfig] = None)` | 150 | Initialize content filter. |
| ⚙️ | `check` | `(self, tool_name: str, arguments: dict[str, Any]) -> GuardrailResult` | 176 | Check arguments for blocked content. |
| ⚙️ | `_arguments_to_text` | `(self, arguments: dict[str, Any]) -> str` | 213 | Convert arguments to searchable text. |

### 🏛️ `class PolicyEnforcer(Guardrail)`  <sub>line 221</sub>

> Enforces tool-specific policies.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, allowed_tools: Optional[list[str]] = None, blocked_tools: Optional[list[str]] = None, tool_limits: Optional[dict[str, int]] = None, config: Optional[GuardrailConfig] = None)` | 236 | Initialize policy enforcer. |
| ⚙️ | `check` | `(self, tool_name: str, arguments: dict[str, Any]) -> GuardrailResult` | 257 | Check if tool execution is allowed by policy. |
| ⚙️ | `reset_counts` | `(self) -> None` | 298 | Reset call counts for all tools. |
| ⚙️ | `get_counts` | `(self) -> dict[str, int]` | 302 | Get current call counts. |

### 🏛️ `class RateLimiter(Guardrail)`  <sub>line 307</sub>

> Rate limiting guardrail.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, calls_per_minute: int = 60, calls_per_hour: int = 1000, config: Optional[GuardrailConfig] = None)` | 322 | Initialize rate limiter. |
| ⚙️ | `check` | `(self, tool_name: str, arguments: dict[str, Any]) -> GuardrailResult` | 341 | Check if within rate limits. |

### 🏛️ `class InputValidator(Guardrail)`  <sub>line 389</sub>

> Validates tool inputs against schemas.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, required_fields: Optional[dict[str, list[str]]] = None, max_lengths: Optional[dict[str, int]] = None, config: Optional[GuardrailConfig] = None)` | 402 | Initialize input validator. |
| ⚙️ | `check` | `(self, tool_name: str, arguments: dict[str, Any]) -> GuardrailResult` | 419 | Validate inputs. |

---

## `argus/tools/integrations/__init__.py`

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `list_all_tools` | `() -> list[str]` | 253 | List all available integration tools. |
| 🔧 | `list_tool_categories` | `() -> list[str]` | 258 | List available tool categories. |
| 🔧 | `get_tools_by_category` | `(category: str) -> list` | 263 | Get tool classes for a specific category. |
| 🔧 | `get_all_tools` | `()` | 270 | Get instances of all tools. |
| 🔧 | `get_tool_count` | `() -> int` | 279 | Get total count of available tools. |

---

## `argus/tools/integrations/ai_agents/agentmail.py`

### 🏛️ `class EmailMessage()`  <sub>line 28</sub>

> Decorators: `@dataclass`  
> Represents an email message.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 39 |  |

### 🏛️ `class AgentInbox()`  <sub>line 53</sub>

> Decorators: `@dataclass`  
> Virtual inbox for an AI agent.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `add_message` | `(self, msg: EmailMessage) -> None` | 61 |  |
| ⚙️ | `get_unread` | `(self) -> list[EmailMessage]` | 64 |  |
| ⚙️ | `mark_read` | `(self, message_id: str) -> bool` | 67 |  |

### 🏛️ `class AgentMailTool(BaseTool)`  <sub>line 75</sub>

> AgentMail - Email management system for AI agents.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, smtp_host: Optional[str] = None, smtp_port: int = 587, smtp_user: Optional[str] = None, smtp_password: Optional[str] = None, imap_host: Optional[str] = None, imap_port: int = 993, domain: str = 'agent.local', use_real_smtp: bool = False, config: Optional[ToolConfig] = None)` | 95 |  |
| ⚙️ | `execute` | `(self, action: str = 'list_inboxes', inbox_id: Optional[str] = None, agent_name: Optional[str] = None, to: Optional[str] = None, subject: Optional[str] = None, body: Optional[str] = None, message_id: Optional[str] = None, limit: int = 20, **kwargs: Any) -> ToolResult` | 127 | Execute AgentMail operations. |
| ⚙️ | `_create_inbox` | `(self, agent_name: Optional[str] = None, **kwargs) -> ToolResult` | 188 | Create a new inbox for an agent. |
| ⚙️ | `_delete_inbox` | `(self, inbox_id: Optional[str] = None, **kwargs) -> ToolResult` | 216 | Delete an inbox. |
| ⚙️ | `_list_inboxes` | `(self, **kwargs) -> ToolResult` | 228 | List all agent inboxes. |
| ⚙️ | `_send_email` | `(self, inbox_id: Optional[str] = None, to: Optional[str] = None, subject: Optional[str] = None, body: Optional[str] = None, **kwargs) -> ToolResult` | 243 | Send an email from an agent inbox. |
| ⚙️ | `_receive_emails` | `(self, inbox_id: Optional[str] = None, limit: int = 20, unread_only: bool = False, **kwargs) -> ToolResult` | 294 | Receive emails for an inbox. |
| ⚙️ | `_get_message` | `(self, inbox_id: Optional[str] = None, message_id: Optional[str] = None, **kwargs) -> ToolResult` | 360 | Get a specific message by ID. |
| ⚙️ | `_mark_read` | `(self, inbox_id: Optional[str] = None, message_id: Optional[str] = None, **kwargs) -> ToolResult` | 379 | Mark a message as read. |
| ⚙️ | `_search_messages` | `(self, inbox_id: Optional[str] = None, query: Optional[str] = None, limit: int = 20, **kwargs) -> ToolResult` | 396 | Search messages in an inbox. |
| ⚙️ | `_get_stats` | `(self, inbox_id: Optional[str] = None, **kwargs) -> ToolResult` | 424 | Get statistics for an inbox or all inboxes. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 451 |  |

---

## `argus/tools/integrations/ai_agents/agentops.py`

### 🏛️ `class AgentEvent()`  <sub>line 26</sub>

> Decorators: `@dataclass`  
> Represents a single agent event.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 38 |  |

### 🏛️ `class AgentSession()`  <sub>line 53</sub>

> Decorators: `@dataclass`  
> Represents an agent session for replay and analysis.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `is_active` | `(self) -> bool` | 65 |  |
| ⚙️ `@property` | `duration_seconds` | `(self) -> float` | 69 |  |
| ⚙️ | `add_event` | `(self, event: AgentEvent) -> None` | 73 |  |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 76 |  |

### 🏛️ `class AgentOpsTool(BaseTool)`  <sub>line 91</sub>

> AgentOps - Comprehensive observability for AI agents.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, api_url: str = 'https://api.agentops.ai', enable_cloud_sync: bool = False, config: Optional[ToolConfig] = None)` | 116 |  |
| ⚙️ | `execute` | `(self, action: str = 'get_metrics', session_id: Optional[str] = None, agent_id: Optional[str] = None, agent_name: Optional[str] = None, event_type: Optional[str] = None, data: Optional[dict] = None, tags: Optional[list[str]] = None, metadata: Optional[dict] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 136 | Execute AgentOps operations. |
| ⚙️ | `_register_agent` | `(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None, metadata: Optional[dict] = None, **kwargs) -> ToolResult` | 200 | Register an agent for tracking. |
| ⚙️ | `_start_session` | `(self, agent_name: Optional[str] = None, agent_id: Optional[str] = None, tags: Optional[list[str]] = None, metadata: Optional[dict] = None, **kwargs) -> ToolResult` | 228 | Start a new agent session. |
| ⚙️ | `_end_session` | `(self, session_id: Optional[str] = None, **kwargs) -> ToolResult` | 273 | End an agent session. |
| ⚙️ | `_track_event_internal` | `(self, session: AgentSession, event_type: str, data: dict, duration_ms: Optional[float] = None, success: bool = True, error: Optional[str] = None) -> AgentEvent` | 303 | Internal method to track an event. |
| ⚙️ | `_track_event` | `(self, session_id: Optional[str] = None, event_type: Optional[str] = None, data: Optional[dict] = None, duration_ms: Optional[float] = None, success: bool = True, error: Optional[str] = None, **kwargs) -> ToolResult` | 327 | Track an event in a session. |
| ⚙️ | `_get_replay` | `(self, session_id: Optional[str] = None, **kwargs) -> ToolResult` | 370 | Get full session replay data. |
| ⚙️ | `_list_sessions` | `(self, agent_id: Optional[str] = None, tags: Optional[list[str]] = None, active_only: bool = False, limit: int = 100, **kwargs) -> ToolResult` | 397 | List sessions with optional filtering. |
| ⚙️ | `_get_session` | `(self, session_id: Optional[str] = None, **kwargs) -> ToolResult` | 424 | Get details for a specific session. |
| ⚙️ | `_get_metrics` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 436 | Get aggregated metrics. |
| ⚙️ | `_get_agent_stats` | `(self, agent_id: Optional[str] = None, **kwargs) -> ToolResult` | 457 | Get statistics for an agent or all agents. |
| ⚙️ | `_search_events` | `(self, session_id: Optional[str] = None, event_type: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 488 | Search events across sessions. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 513 |  |

---

## `argus/tools/integrations/ai_agents/freeplay.py`

### 🏛️ `class PromptTemplate()`  <sub>line 24</sub>

> Decorators: `@dataclass`  
> A prompt template with version tracking.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `render` | `(self, **kwargs) -> str` | 34 | Render template with variables. |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 42 |  |

### 🏛️ `class Experiment()`  <sub>line 55</sub>

> Decorators: `@dataclass`  
> An A/B test experiment for prompts.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `add_result` | `(self, variant_id: str, score: float) -> None` | 64 |  |
| ⚙️ | `get_winner` | `(self) -> Optional[str]` | 67 |  |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 75 |  |

### 🏛️ `class EvaluationRun()`  <sub>line 88</sub>

> Decorators: `@dataclass`  
> An evaluation run for measuring agent performance.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 99 |  |

### 🏛️ `class FreeplayTool(BaseTool)`  <sub>line 112</sub>

> Freeplay - Build, optimize, and evaluate AI agents.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, project_id: Optional[str] = None, config: Optional[ToolConfig] = None)` | 137 |  |
| ⚙️ | `execute` | `(self, action: str = 'list_templates', name: Optional[str] = None, template: Optional[str] = None, template_id: Optional[str] = None, variables: Optional[dict] = None, variants: Optional[list] = None, experiment_id: Optional[str] = None, test_cases: Optional[list] = None, run_id: Optional[str] = None, score: Optional[float] = None, variant_id: Optional[str] = None, **kwargs: Any) -> ToolResult` | 156 | Execute Freeplay operations. |
| ⚙️ | `_create_template` | `(self, name: Optional[str] = None, template: Optional[str] = None, **kwargs) -> ToolResult` | 219 | Create a new prompt template. |
| ⚙️ | `_update_template` | `(self, template_id: Optional[str] = None, template: Optional[str] = None, **kwargs) -> ToolResult` | 252 | Update an existing template (creates new version). |
| ⚙️ | `_render_template` | `(self, template_id: Optional[str] = None, variables: Optional[dict] = None, **kwargs) -> ToolResult` | 280 | Render a template with variables. |
| ⚙️ | `_list_templates` | `(self, **kwargs) -> ToolResult` | 299 | List all templates. |
| ⚙️ | `_get_template` | `(self, template_id: Optional[str] = None, **kwargs) -> ToolResult` | 304 | Get a specific template. |
| ⚙️ | `_create_experiment` | `(self, name: Optional[str] = None, variants: Optional[list] = None, **kwargs) -> ToolResult` | 315 | Create a new A/B experiment. |
| ⚙️ | `_record_result` | `(self, experiment_id: Optional[str] = None, variant_id: Optional[str] = None, score: Optional[float] = None, **kwargs) -> ToolResult` | 349 | Record a result for an experiment variant. |
| ⚙️ | `_get_experiment` | `(self, experiment_id: Optional[str] = None, **kwargs) -> ToolResult` | 374 | Get experiment details and results. |
| ⚙️ | `_list_experiments` | `(self, **kwargs) -> ToolResult` | 385 | List all experiments. |
| ⚙️ | `_create_evaluation` | `(self, name: Optional[str] = None, test_cases: Optional[list] = None, **kwargs) -> ToolResult` | 390 | Create a new evaluation run. |
| ⚙️ | `_run_evaluation` | `(self, run_id: Optional[str] = None, evaluator_fn: Optional[str] = None, **kwargs) -> ToolResult` | 416 | Run an evaluation (simulated). |
| ⚙️ | `_get_evaluation` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 459 | Get evaluation details. |
| ⚙️ | `_list_evaluations` | `(self, **kwargs) -> ToolResult` | 470 | List all evaluations. |
| ⚙️ | `_log_trace` | `(self, trace_data: Optional[dict] = None, **kwargs) -> ToolResult` | 475 | Log a trace event. |
| ⚙️ | `_get_traces` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 494 | Get recent traces. |
| ⚙️ | `_get_analytics` | `(self, **kwargs) -> ToolResult` | 503 | Get analytics summary. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 514 |  |

---

## `argus/tools/integrations/ai_agents/goodmem.py`

### 🏛️ `class MemoryEntry()`  <sub>line 26</sub>

> Decorators: `@dataclass`  
> Represents a single memory entry.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 39 |  |

### 🏛️ `class MemorySpace()`  <sub>line 54</sub>

> Decorators: `@dataclass`  
> A namespace for organizing agent memories.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `add_memory` | `(self, memory: MemoryEntry) -> None` | 62 |  |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 65 |  |

### 🏛️ `class GoodMemTool(BaseTool)`  <sub>line 75</sub>

> GoodMem - Persistent semantic memory for AI agents.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, storage_path: Optional[str] = None, enable_persistence: bool = True, embedding_model: str = 'sentence-transformers', config: Optional[ToolConfig] = None)` | 99 |  |
| ⚙️ | `_get_embedder` | `(self)` | 124 | Lazy-load the embedding model. |
| ⚙️ | `_compute_embedding` | `(self, text: str) -> list[float]` | 135 | Compute embedding for text. |
| ⚙️ | `_cosine_similarity` | `(self, a: list[float], b: list[float]) -> float` | 144 | Compute cosine similarity between two vectors. |
| ⚙️ | `_load_memories` | `(self) -> None` | 154 | Load memories from persistent storage. |
| ⚙️ | `_save_memories` | `(self) -> None` | 192 | Save memories to persistent storage. |
| ⚙️ | `execute` | `(self, action: str = 'search', agent_id: Optional[str] = None, space_name: str = 'default', content: Optional[str] = None, query: Optional[str] = None, memory_id: Optional[str] = None, tags: Optional[list[str]] = None, metadata: Optional[dict] = None, importance: float = 0.5, limit: int = 10, threshold: float = 0.3, **kwargs: Any) -> ToolResult` | 216 | Execute GoodMem operations. |
| ⚙️ | `_get_or_create_space` | `(self, agent_id: str, space_name: str) -> MemorySpace` | 292 | Get or create a memory space. |
| ⚙️ | `_store_memory` | `(self, agent_id: Optional[str] = None, space_name: str = 'default', content: Optional[str] = None, tags: Optional[list[str]] = None, metadata: Optional[dict] = None, importance: float = 0.5, **kwargs) -> ToolResult` | 306 | Store a new memory. |
| ⚙️ | `_search_memories` | `(self, agent_id: Optional[str] = None, space_name: str = 'default', query: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, threshold: float = 0.3, **kwargs) -> ToolResult` | 348 | Search memories using semantic similarity. |
| ⚙️ | `_recall_memory` | `(self, memory_id: Optional[str] = None, agent_id: Optional[str] = None, **kwargs) -> ToolResult` | 404 | Recall a specific memory by ID. |
| ⚙️ | `_forget_memory` | `(self, memory_id: Optional[str] = None, agent_id: Optional[str] = None, **kwargs) -> ToolResult` | 426 | Forget (delete) a memory. |
| ⚙️ | `_list_memories` | `(self, agent_id: Optional[str] = None, space_name: str = 'default', limit: int = 20, **kwargs) -> ToolResult` | 447 | List memories in a space. |
| ⚙️ | `_create_space` | `(self, agent_id: Optional[str] = None, space_name: str = 'default', **kwargs) -> ToolResult` | 473 | Create a new memory space. |
| ⚙️ | `_list_spaces` | `(self, agent_id: Optional[str] = None, **kwargs) -> ToolResult` | 491 | List memory spaces. |
| ⚙️ | `_consolidate_memories` | `(self, agent_id: Optional[str] = None, space_name: str = 'default', similarity_threshold: float = 0.85, **kwargs) -> ToolResult` | 508 | Consolidate similar memories to reduce redundancy. |
| ⚙️ | `_get_stats` | `(self, agent_id: Optional[str] = None, **kwargs) -> ToolResult` | 559 | Get memory statistics. |
| ⚙️ | `_update_importance` | `(self, memory_id: Optional[str] = None, importance: float = 0.5, **kwargs) -> ToolResult` | 589 | Update memory importance score. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 610 |  |

---

## `argus/tools/integrations/cloud/bigquery.py`

### 🏛️ `class QueryResult()`  <sub>line 23</sub>

> Decorators: `@dataclass`  
> Result of a BigQuery query.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 33 |  |

### 🏛️ `class BigQueryTool(BaseTool)`  <sub>line 45</sub>

> BigQuery - Data analysis and querying tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None, location: str = 'US', config: Optional[ToolConfig] = None)` | 67 |  |
| ⚙️ | `_get_client` | `(self)` | 83 | Lazy-load BigQuery client. |
| ⚙️ | `execute` | `(self, action: str = 'query', sql: Optional[str] = None, dataset: Optional[str] = None, table: Optional[str] = None, limit: int = 100, dry_run: bool = False, parameters: Optional[dict] = None, **kwargs: Any) -> ToolResult` | 93 | Execute BigQuery operations. |
| ⚙️ | `_execute_query` | `(self, sql: Optional[str] = None, limit: int = 100, dry_run: bool = False, parameters: Optional[dict] = None, **kwargs) -> ToolResult` | 152 | Execute a SQL query. |
| ⚙️ | `_list_datasets` | `(self, **kwargs) -> ToolResult` | 238 | List all datasets in the project. |
| ⚙️ | `_list_tables` | `(self, dataset: Optional[str] = None, **kwargs) -> ToolResult` | 256 | List tables in a dataset. |
| ⚙️ | `_get_schema` | `(self, dataset: Optional[str] = None, table: Optional[str] = None, **kwargs) -> ToolResult` | 281 | Get table schema. |
| ⚙️ | `_preview_table` | `(self, dataset: Optional[str] = None, table: Optional[str] = None, limit: int = 10, **kwargs) -> ToolResult` | 313 | Preview table data. |
| ⚙️ | `_create_dataset` | `(self, dataset: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 327 | Create a new dataset. |
| ⚙️ | `_delete_dataset` | `(self, dataset: Optional[str] = None, delete_contents: bool = False, **kwargs) -> ToolResult` | 354 | Delete a dataset. |
| ⚙️ | `_analyze_table` | `(self, dataset: Optional[str] = None, table: Optional[str] = None, **kwargs) -> ToolResult` | 373 | Analyze table statistics. |
| ⚙️ | `_estimate_cost` | `(self, sql: Optional[str] = None, **kwargs) -> ToolResult` | 416 | Estimate query cost. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 424 |  |

---

## `argus/tools/integrations/cloud/cloud_trace.py`

### 🏛️ `class Span()`  <sub>line 25</sub>

> Decorators: `@dataclass`  
> A trace span representing a unit of work.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `duration_ms` | `(self) -> float` | 38 |  |
| ⚙️ | `add_event` | `(self, name: str, attributes: Optional[dict] = None) -> None` | 43 |  |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 50 |  |

### 🏛️ `class Trace()`  <sub>line 66</sub>

> Decorators: `@dataclass`  
> A complete trace representing a request lifecycle.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ `@property` | `root_span` | `(self) -> Optional[Span]` | 74 |  |
| ⚙️ `@property` | `duration_ms` | `(self) -> float` | 81 |  |
| ⚙️ | `to_dict` | `(self) -> dict[str, Any]` | 86 |  |

### 🏛️ `class CloudTraceTool(BaseTool)`  <sub>line 97</sub>

> Google Cloud Trace - Distributed tracing for AI agents.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, project_id: Optional[str] = None, enable_cloud_export: bool = False, config: Optional[ToolConfig] = None)` | 121 |  |
| ⚙️ | `_get_trace_client` | `(self)` | 142 | Lazy-load Cloud Trace client. |
| ⚙️ | `execute` | `(self, action: str = 'list_traces', trace_id: Optional[str] = None, span_id: Optional[str] = None, name: Optional[str] = None, attributes: Optional[dict] = None, status: Optional[str] = None, event_name: Optional[str] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 152 | Execute Cloud Trace operations. |
| ⚙️ | `_start_trace` | `(self, name: Optional[str] = None, attributes: Optional[dict] = None, **kwargs) -> ToolResult` | 200 | Start a new trace. |
| ⚙️ | `_end_trace` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 238 | End a trace. |
| ⚙️ | `_start_span` | `(self, trace_id: Optional[str] = None, name: Optional[str] = None, parent_span_id: Optional[str] = None, attributes: Optional[dict] = None, **kwargs) -> ToolResult` | 269 | Start a new span within a trace. |
| ⚙️ | `_end_span` | `(self, span_id: Optional[str] = None, status: Optional[str] = None, **kwargs) -> ToolResult` | 308 | End a span. |
| ⚙️ | `_add_event` | `(self, span_id: Optional[str] = None, event_name: Optional[str] = None, attributes: Optional[dict] = None, **kwargs) -> ToolResult` | 336 | Add an event to a span. |
| ⚙️ | `_set_status` | `(self, span_id: Optional[str] = None, status: Optional[str] = None, **kwargs) -> ToolResult` | 358 | Set span status. |
| ⚙️ | `_set_attributes` | `(self, span_id: Optional[str] = None, attributes: Optional[dict] = None, **kwargs) -> ToolResult` | 378 | Set span attributes. |
| ⚙️ | `_get_trace` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 396 | Get trace details. |
| ⚙️ | `_get_span` | `(self, span_id: Optional[str] = None, **kwargs) -> ToolResult` | 408 | Get span details. |
| ⚙️ | `_list_traces` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 420 | List all traces. |
| ⚙️ | `_export_trace` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 442 | Export a trace to Cloud Trace. |
| ⚙️ | `_export_to_cloud` | `(self, trace: Trace) -> None` | 471 | Export trace to Google Cloud Trace. |
| ⚙️ | `_get_active_span` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 497 | Get the currently active span for a trace. |
| ⚙️ `@contextmanager` | `trace_context` | `(self, name: str, attributes: Optional[dict] = None)` | 513 | Context manager for tracing a code block. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 529 |  |

---

## `argus/tools/integrations/cloud/pubsub.py`

### 🏛️ `class PubSubTool(BaseTool)`  <sub>line 21</sub>

> Google Cloud Pub/Sub - Message queue operations.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None, config: Optional[ToolConfig] = None)` | 42 |  |
| ⚙️ | `_get_publisher` | `(self)` | 57 | Lazy-load publisher client. |
| ⚙️ | `_get_subscriber` | `(self)` | 67 | Lazy-load subscriber client. |
| ⚙️ | `execute` | `(self, action: str = 'list_topics', topic: Optional[str] = None, subscription: Optional[str] = None, message: Optional[str] = None, messages: Optional[list] = None, attributes: Optional[dict] = None, ack_ids: Optional[list] = None, max_messages: int = 10, **kwargs: Any) -> ToolResult` | 77 | Execute Pub/Sub operations. |
| ⚙️ | `_topic_path` | `(self, topic: str) -> str` | 142 | Get full topic path. |
| ⚙️ | `_subscription_path` | `(self, subscription: str) -> str` | 146 | Get full subscription path. |
| ⚙️ | `_publish` | `(self, topic: Optional[str] = None, message: Optional[str] = None, attributes: Optional[dict] = None, **kwargs) -> ToolResult` | 150 | Publish a message to a topic. |
| ⚙️ | `_publish_batch` | `(self, topic: Optional[str] = None, messages: Optional[list] = None, **kwargs) -> ToolResult` | 186 | Publish multiple messages in batch. |
| ⚙️ | `_pull` | `(self, subscription: Optional[str] = None, max_messages: int = 10, **kwargs) -> ToolResult` | 226 | Pull messages from a subscription. |
| ⚙️ | `_acknowledge` | `(self, subscription: Optional[str] = None, ack_ids: Optional[list] = None, **kwargs) -> ToolResult` | 265 | Acknowledge messages. |
| ⚙️ | `_list_topics` | `(self, **kwargs) -> ToolResult` | 292 | List all topics. |
| ⚙️ | `_list_subscriptions` | `(self, topic: Optional[str] = None, **kwargs) -> ToolResult` | 309 | List subscriptions. |
| ⚙️ | `_create_topic` | `(self, topic: Optional[str] = None, **kwargs) -> ToolResult` | 334 | Create a new topic. |
| ⚙️ | `_delete_topic` | `(self, topic: Optional[str] = None, **kwargs) -> ToolResult` | 354 | Delete a topic. |
| ⚙️ | `_create_subscription` | `(self, subscription: Optional[str] = None, topic: Optional[str] = None, ack_deadline: int = 10, **kwargs) -> ToolResult` | 373 | Create a subscription. |
| ⚙️ | `_delete_subscription` | `(self, subscription: Optional[str] = None, **kwargs) -> ToolResult` | 405 | Delete a subscription. |
| ⚙️ | `_get_topic` | `(self, topic: Optional[str] = None, **kwargs) -> ToolResult` | 424 | Get topic details. |
| ⚙️ | `_get_subscription` | `(self, subscription: Optional[str] = None, **kwargs) -> ToolResult` | 443 | Get subscription details. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 465 |  |

---

## `argus/tools/integrations/cloud/vertex_ai.py`

### 🏛️ `class VertexAISearchTool(BaseTool)`  <sub>line 21</sub>

> Vertex AI Search - Search across private data stores.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, project_id: Optional[str] = None, location: str = 'global', datastore_id: Optional[str] = None, engine_id: Optional[str] = None, config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_client` | `(self)` | 59 | Lazy-load search client. |
| ⚙️ | `execute` | `(self, action: str = 'search', query: Optional[str] = None, page_size: int = 10, page_token: Optional[str] = None, filter_str: Optional[str] = None, order_by: Optional[str] = None, boost_spec: Optional[dict] = None, **kwargs: Any) -> ToolResult` | 69 | Execute Vertex AI Search operations. |
| ⚙️ | `_search` | `(self, query: Optional[str] = None, page_size: int = 10, page_token: Optional[str] = None, filter_str: Optional[str] = None, **kwargs) -> ToolResult` | 109 | Search the data store. |
| ⚙️ | `_list_datastores` | `(self, **kwargs) -> ToolResult` | 163 | List available data stores. |
| ⚙️ | `_get_datastore` | `(self, datastore_id: Optional[str] = None, **kwargs) -> ToolResult` | 187 | Get data store details. |
| ⚙️ | `_get_document` | `(self, document_id: Optional[str] = None, datastore_id: Optional[str] = None, **kwargs) -> ToolResult` | 213 | Get a specific document. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 244 |  |

### 🏛️ `class VertexAIRAGTool(BaseTool)`  <sub>line 265</sub>

> Vertex AI RAG Engine - Retrieval Augmented Generation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, project_id: Optional[str] = None, location: str = 'us-central1', config: Optional[ToolConfig] = None)` | 285 |  |
| ⚙️ | `execute` | `(self, action: str = 'list_corpora', corpus_name: Optional[str] = None, query: Optional[str] = None, top_k: int = 10, file_path: Optional[str] = None, document_name: Optional[str] = None, **kwargs: Any) -> ToolResult` | 298 | Execute Vertex AI RAG operations. |
| ⚙️ | `_list_corpora` | `(self, **kwargs) -> ToolResult` | 338 | List RAG corpora. |
| ⚙️ | `_create_corpus` | `(self, corpus_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 362 | Create a new RAG corpus. |
| ⚙️ | `_delete_corpus` | `(self, corpus_name: Optional[str] = None, **kwargs) -> ToolResult` | 388 | Delete a RAG corpus. |
| ⚙️ | `_retrieve` | `(self, corpus_name: Optional[str] = None, query: Optional[str] = None, top_k: int = 10, **kwargs) -> ToolResult` | 409 | Retrieve relevant documents from corpus. |
| ⚙️ | `_import_files` | `(self, corpus_name: Optional[str] = None, file_path: Optional[str] = None, gcs_uri: Optional[str] = None, **kwargs) -> ToolResult` | 447 | Import files into a RAG corpus. |
| ⚙️ | `_list_files` | `(self, corpus_name: Optional[str] = None, **kwargs) -> ToolResult` | 480 | List files in a corpus. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 510 |  |

---

## `argus/tools/integrations/communication/mailgun.py`

### 🏛️ `class MailgunTool(BaseTool)`  <sub>line 19</sub>

> Mailgun - Email sending and management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, domain: Optional[str] = None, config: Optional[ToolConfig] = None)` | 42 |  |
| ⚙️ | `_get_session` | `(self)` | 60 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None, files: Optional[dict] = None) -> dict` | 74 | Make API request to Mailgun. |
| ⚙️ | `execute` | `(self, action: str = 'send', to: Optional[str] = None, to_list: Optional[list] = None, from_email: Optional[str] = None, subject: Optional[str] = None, text: Optional[str] = None, html: Optional[str] = None, cc: Optional[str] = None, bcc: Optional[str] = None, reply_to: Optional[str] = None, template: Optional[str] = None, template_variables: Optional[dict] = None, tags: Optional[list] = None, list_address: Optional[str] = None, member_email: Optional[str] = None, member_name: Optional[str] = None, event_type: Optional[str] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 103 | Execute Mailgun operations. |
| ⚙️ | `_send_email` | `(self, to: Optional[str] = None, to_list: Optional[list] = None, from_email: Optional[str] = None, subject: Optional[str] = None, text: Optional[str] = None, html: Optional[str] = None, cc: Optional[str] = None, bcc: Optional[str] = None, reply_to: Optional[str] = None, tags: Optional[list] = None, **kwargs) -> ToolResult` | 186 | Send an email. |
| ⚙️ | `_send_template` | `(self, to: Optional[str] = None, to_list: Optional[list] = None, from_email: Optional[str] = None, subject: Optional[str] = None, template: Optional[str] = None, template_variables: Optional[dict] = None, **kwargs) -> ToolResult` | 239 | Send an email using a template. |
| ⚙️ | `_list_mailing_lists` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 279 | List all mailing lists. |
| ⚙️ | `_create_mailing_list` | `(self, list_address: Optional[str] = None, **kwargs) -> ToolResult` | 302 | Create a mailing list. |
| ⚙️ | `_get_mailing_list` | `(self, list_address: Optional[str] = None, **kwargs) -> ToolResult` | 330 | Get mailing list details. |
| ⚙️ | `_delete_mailing_list` | `(self, list_address: Optional[str] = None, **kwargs) -> ToolResult` | 343 | Delete a mailing list. |
| ⚙️ | `_add_member` | `(self, list_address: Optional[str] = None, member_email: Optional[str] = None, member_name: Optional[str] = None, **kwargs) -> ToolResult` | 359 | Add a member to a mailing list. |
| ⚙️ | `_list_members` | `(self, list_address: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 391 | List members of a mailing list. |
| ⚙️ | `_remove_member` | `(self, list_address: Optional[str] = None, member_email: Optional[str] = None, **kwargs) -> ToolResult` | 422 | Remove a member from a mailing list. |
| ⚙️ | `_get_events` | `(self, event_type: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 443 | Get email events. |
| ⚙️ | `_list_domains` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 481 | List all domains. |
| ⚙️ | `_get_domain` | `(self, **kwargs) -> ToolResult` | 504 | Get domain details. |
| ⚙️ | `_verify_domain` | `(self, **kwargs) -> ToolResult` | 518 | Verify domain DNS records. |
| ⚙️ | `_get_stats` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 532 | Get email statistics. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 554 |  |

---

## `argus/tools/integrations/communication/paypal.py`

### 🏛️ `class PayPalTool(BaseTool)`  <sub>line 19</sub>

> PayPal - Payment processing and order management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, sandbox: bool = True, config: Optional[ToolConfig] = None)` | 44 |  |
| ⚙️ | `_get_access_token` | `(self) -> str` | 67 | Get OAuth access token. |
| ⚙️ | `_get_session` | `(self)` | 93 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 106 | Make API request to PayPal. |
| ⚙️ | `execute` | `(self, action: str = 'list_orders', order_id: Optional[str] = None, capture_id: Optional[str] = None, payment_id: Optional[str] = None, payout_id: Optional[str] = None, subscription_id: Optional[str] = None, plan_id: Optional[str] = None, amount: Optional[str] = None, currency: str = 'USD', description: Optional[str] = None, return_url: Optional[str] = None, cancel_url: Optional[str] = None, items: Optional[list] = None, recipient_email: Optional[str] = None, **kwargs: Any) -> ToolResult` | 137 | Execute PayPal operations. |
| ⚙️ | `_create_order` | `(self, amount: Optional[str] = None, currency: str = 'USD', description: Optional[str] = None, return_url: Optional[str] = None, cancel_url: Optional[str] = None, items: Optional[list] = None, **kwargs) -> ToolResult` | 215 | Create an order. |
| ⚙️ | `_get_order` | `(self, order_id: Optional[str] = None, **kwargs) -> ToolResult` | 286 | Get order details. |
| ⚙️ | `_capture_order` | `(self, order_id: Optional[str] = None, **kwargs) -> ToolResult` | 307 | Capture payment for an order. |
| ⚙️ | `_authorize_order` | `(self, order_id: Optional[str] = None, **kwargs) -> ToolResult` | 331 | Authorize payment for an order. |
| ⚙️ | `_refund_capture` | `(self, capture_id: Optional[str] = None, amount: Optional[str] = None, currency: str = 'USD', **kwargs) -> ToolResult` | 349 | Refund a captured payment. |
| ⚙️ | `_get_capture` | `(self, capture_id: Optional[str] = None, **kwargs) -> ToolResult` | 379 | Get capture details. |
| ⚙️ | `_create_payout` | `(self, recipient_email: Optional[str] = None, amount: Optional[str] = None, currency: str = 'USD', **kwargs) -> ToolResult` | 393 | Create a payout. |
| ⚙️ | `_get_payout` | `(self, payout_id: Optional[str] = None, **kwargs) -> ToolResult` | 436 | Get payout details. |
| ⚙️ | `_list_products` | `(self, **kwargs) -> ToolResult` | 456 | List products. |
| ⚙️ | `_create_product` | `(self, description: Optional[str] = None, **kwargs) -> ToolResult` | 475 | Create a product. |
| ⚙️ | `_get_product` | `(self, **kwargs) -> ToolResult` | 501 | Get product details. |
| ⚙️ | `_list_plans` | `(self, **kwargs) -> ToolResult` | 512 | List subscription plans. |
| ⚙️ | `_create_plan` | `(self, amount: Optional[str] = None, currency: str = 'USD', description: Optional[str] = None, **kwargs) -> ToolResult` | 531 | Create a subscription plan. |
| ⚙️ | `_get_plan` | `(self, plan_id: Optional[str] = None, **kwargs) -> ToolResult` | 588 | Get plan details. |
| ⚙️ | `_create_subscription` | `(self, plan_id: Optional[str] = None, return_url: Optional[str] = None, cancel_url: Optional[str] = None, **kwargs) -> ToolResult` | 602 | Create a subscription. |
| ⚙️ | `_get_subscription` | `(self, subscription_id: Optional[str] = None, **kwargs) -> ToolResult` | 637 | Get subscription details. |
| ⚙️ | `_cancel_subscription` | `(self, subscription_id: Optional[str] = None, **kwargs) -> ToolResult` | 658 | Cancel a subscription. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 680 |  |

---

## `argus/tools/integrations/communication/stripe_tool.py`

### 🏛️ `class StripeTool(BaseTool)`  <sub>line 18</sub>

> Stripe - Payment processing and subscription management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 43 |  |
| ⚙️ | `_get_session` | `(self)` | 59 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 70 | Make API request to Stripe. |
| ⚙️ | `execute` | `(self, action: str = 'list_customers', customer_id: Optional[str] = None, email: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None, amount: Optional[int] = None, currency: str = 'usd', payment_method_id: Optional[str] = None, payment_intent_id: Optional[str] = None, product_id: Optional[str] = None, price_id: Optional[str] = None, subscription_id: Optional[str] = None, invoice_id: Optional[str] = None, metadata: Optional[dict] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 98 | Execute Stripe operations. |
| ⚙️ | `_list_customers` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 192 | List customers. |
| ⚙️ | `_get_customer` | `(self, customer_id: Optional[str] = None, **kwargs) -> ToolResult` | 216 | Get customer details. |
| ⚙️ | `_create_customer` | `(self, email: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None, metadata: Optional[dict] = None, **kwargs) -> ToolResult` | 239 | Create a customer. |
| ⚙️ | `_update_customer` | `(self, customer_id: Optional[str] = None, email: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 268 | Update a customer. |
| ⚙️ | `_delete_customer` | `(self, customer_id: Optional[str] = None, **kwargs) -> ToolResult` | 298 | Delete a customer. |
| ⚙️ | `_list_payment_intents` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 315 | List payment intents. |
| ⚙️ | `_get_payment_intent` | `(self, payment_intent_id: Optional[str] = None, **kwargs) -> ToolResult` | 342 | Get payment intent details. |
| ⚙️ | `_create_payment_intent` | `(self, amount: Optional[int] = None, currency: str = 'usd', customer_id: Optional[str] = None, payment_method_id: Optional[str] = None, metadata: Optional[dict] = None, **kwargs) -> ToolResult` | 365 | Create a payment intent. |
| ⚙️ | `_confirm_payment_intent` | `(self, payment_intent_id: Optional[str] = None, payment_method_id: Optional[str] = None, **kwargs) -> ToolResult` | 400 | Confirm a payment intent. |
| ⚙️ | `_cancel_payment_intent` | `(self, payment_intent_id: Optional[str] = None, **kwargs) -> ToolResult` | 426 | Cancel a payment intent. |
| ⚙️ | `_list_subscriptions` | `(self, customer_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 447 | List subscriptions. |
| ⚙️ | `_get_subscription` | `(self, subscription_id: Optional[str] = None, **kwargs) -> ToolResult` | 477 | Get subscription details. |
| ⚙️ | `_create_subscription` | `(self, customer_id: Optional[str] = None, price_id: Optional[str] = None, **kwargs) -> ToolResult` | 499 | Create a subscription. |
| ⚙️ | `_update_subscription` | `(self, subscription_id: Optional[str] = None, **kwargs) -> ToolResult` | 524 | Update a subscription. |
| ⚙️ | `_cancel_subscription` | `(self, subscription_id: Optional[str] = None, **kwargs) -> ToolResult` | 554 | Cancel a subscription. |
| ⚙️ | `_list_invoices` | `(self, customer_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 572 | List invoices. |
| ⚙️ | `_get_invoice` | `(self, invoice_id: Optional[str] = None, **kwargs) -> ToolResult` | 603 | Get invoice details. |
| ⚙️ | `_create_invoice` | `(self, customer_id: Optional[str] = None, **kwargs) -> ToolResult` | 616 | Create an invoice. |
| ⚙️ | `_finalize_invoice` | `(self, invoice_id: Optional[str] = None, **kwargs) -> ToolResult` | 635 | Finalize an invoice. |
| ⚙️ | `_pay_invoice` | `(self, invoice_id: Optional[str] = None, **kwargs) -> ToolResult` | 652 | Pay an invoice. |
| ⚙️ | `_list_products` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 670 | List products. |
| ⚙️ | `_get_product` | `(self, product_id: Optional[str] = None, **kwargs) -> ToolResult` | 693 | Get product details. |
| ⚙️ | `_create_product` | `(self, name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 706 | Create a product. |
| ⚙️ | `_list_prices` | `(self, product_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 730 | List prices. |
| ⚙️ | `_get_price` | `(self, price_id: Optional[str] = None, **kwargs) -> ToolResult` | 761 | Get price details. |
| ⚙️ | `_create_price` | `(self, product_id: Optional[str] = None, amount: Optional[int] = None, currency: str = 'usd', **kwargs) -> ToolResult` | 774 | Create a price. |
| ⚙️ | `_get_balance` | `(self, **kwargs) -> ToolResult` | 805 | Get account balance. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 824 |  |

---

## `argus/tools/integrations/database/pandas_tool.py`

### 🏛️ `class PandasTool(BaseTool)`  <sub>line 17</sub>

> Pandas DataFrame analysis tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ToolConfig] = None)` | 31 |  |
| ⚙️ | `execute` | `(self, action: str = 'info', path: Optional[str] = None, query: Optional[str] = None, columns: Optional[list[str]] = None, n: int = 10, **kwargs: Any) -> ToolResult` | 35 | Perform Pandas operation. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 134 |  |

---

## `argus/tools/integrations/database/sql.py`

### 🏛️ `class SqlTool(BaseTool)`  <sub>line 17</sub>

> SQL database query tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, connection_string: Optional[str] = None, read_only: bool = True, config: Optional[ToolConfig] = None)` | 33 |  |
| ⚙️ | `_get_engine` | `(self)` | 44 | Lazy load SQLAlchemy engine. |
| ⚙️ | `_is_read_query` | `(self, query: str) -> bool` | 56 | Check if query is read-only. |
| ⚙️ | `execute` | `(self, query: str = '', params: Optional[dict] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 61 | Execute SQL query. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 119 |  |

---

## `argus/tools/integrations/devops/daytona.py`

### 🏛️ `class DaytonaTool(BaseTool)`  <sub>line 18</sub>

> Daytona - Development environment management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, base_url: Optional[str] = None, config: Optional[ToolConfig] = None)` | 40 |  |
| ⚙️ | `_get_session` | `(self)` | 58 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict \| list` | 71 | Make API request to Daytona. |
| ⚙️ | `execute` | `(self, action: str = 'list_workspaces', workspace_id: Optional[str] = None, project_id: Optional[str] = None, provider_id: Optional[str] = None, target_id: Optional[str] = None, name: Optional[str] = None, repository: Optional[str] = None, branch: Optional[str] = None, **kwargs: Any) -> ToolResult` | 101 | Execute Daytona operations. |
| ⚙️ | `_list_workspaces` | `(self, **kwargs) -> ToolResult` | 169 | List workspaces. |
| ⚙️ | `_get_workspace` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 190 | Get workspace details. |
| ⚙️ | `_create_workspace` | `(self, name: Optional[str] = None, repository: Optional[str] = None, branch: Optional[str] = None, target_id: Optional[str] = None, **kwargs) -> ToolResult` | 212 | Create a workspace. |
| ⚙️ | `_delete_workspace` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 243 | Delete a workspace. |
| ⚙️ | `_start_workspace` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 262 | Start a workspace. |
| ⚙️ | `_stop_workspace` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 278 | Stop a workspace. |
| ⚙️ | `_list_projects` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 295 | List projects in workspace. |
| ⚙️ | `_get_project` | `(self, workspace_id: Optional[str] = None, project_id: Optional[str] = None, **kwargs) -> ToolResult` | 314 | Get project details. |
| ⚙️ | `_create_project` | `(self, workspace_id: Optional[str] = None, name: Optional[str] = None, repository: Optional[str] = None, branch: Optional[str] = None, **kwargs) -> ToolResult` | 332 | Create a project in workspace. |
| ⚙️ | `_delete_project` | `(self, workspace_id: Optional[str] = None, project_id: Optional[str] = None, **kwargs) -> ToolResult` | 362 | Delete a project from workspace. |
| ⚙️ | `_list_providers` | `(self, **kwargs) -> ToolResult` | 383 | List Git providers. |
| ⚙️ | `_get_provider` | `(self, provider_id: Optional[str] = None, **kwargs) -> ToolResult` | 403 | Get Git provider details. |
| ⚙️ | `_add_provider` | `(self, provider_id: Optional[str] = None, **kwargs) -> ToolResult` | 418 | Add a Git provider. |
| ⚙️ | `_remove_provider` | `(self, provider_id: Optional[str] = None, **kwargs) -> ToolResult` | 451 | Remove a Git provider. |
| ⚙️ | `_list_targets` | `(self, **kwargs) -> ToolResult` | 468 | List available targets. |
| ⚙️ | `_get_target` | `(self, target_id: Optional[str] = None, **kwargs) -> ToolResult` | 487 | Get target details. |
| ⚙️ | `_set_target` | `(self, target_id: Optional[str] = None, **kwargs) -> ToolResult` | 502 | Set a target. |
| ⚙️ | `_remove_target` | `(self, target_id: Optional[str] = None, **kwargs) -> ToolResult` | 531 | Remove a target. |
| ⚙️ | `_server_info` | `(self, **kwargs) -> ToolResult` | 548 | Get server information. |
| ⚙️ | `_get_ssh_key` | `(self, **kwargs) -> ToolResult` | 562 | Get SSH public key. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 570 |  |

---

## `argus/tools/integrations/devops/gitlab.py`

### 🏛️ `class GitLabTool(BaseTool)`  <sub>line 19</sub>

> GitLab - Git repository and CI/CD management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, access_token: Optional[str] = None, base_url: str = 'https://gitlab.com', config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_session` | `(self)` | 59 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict \| list` | 70 | Make API request to GitLab. |
| ⚙️ | `_encode_project_path` | `(self, project_id: str) -> str` | 100 | Encode project path for API use. |
| ⚙️ | `execute` | `(self, action: str = 'list_projects', project_id: Optional[str] = None, issue_iid: Optional[int] = None, mr_iid: Optional[int] = None, pipeline_id: Optional[int] = None, job_id: Optional[int] = None, branch: Optional[str] = None, title: Optional[str] = None, description: Optional[str] = None, labels: Optional[list] = None, assignee_ids: Optional[list] = None, source_branch: Optional[str] = None, target_branch: Optional[str] = None, query: Optional[str] = None, state: Optional[str] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 106 | Execute GitLab operations. |
| ⚙️ | `_list_projects` | `(self, query: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 206 | List projects. |
| ⚙️ | `_get_project` | `(self, project_id: Optional[str] = None, **kwargs) -> ToolResult` | 237 | Get project details. |
| ⚙️ | `_create_project` | `(self, title: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 262 | Create a project. |
| ⚙️ | `_delete_project` | `(self, project_id: Optional[str] = None, **kwargs) -> ToolResult` | 289 | Delete a project. |
| ⚙️ | `_list_issues` | `(self, project_id: Optional[str] = None, state: Optional[str] = None, labels: Optional[list] = None, limit: int = 100, **kwargs) -> ToolResult` | 307 | List issues. |
| ⚙️ | `_get_issue` | `(self, project_id: Optional[str] = None, issue_iid: Optional[int] = None, **kwargs) -> ToolResult` | 348 | Get issue details. |
| ⚙️ | `_create_issue` | `(self, project_id: Optional[str] = None, title: Optional[str] = None, description: Optional[str] = None, labels: Optional[list] = None, assignee_ids: Optional[list] = None, **kwargs) -> ToolResult` | 377 | Create an issue. |
| ⚙️ | `_update_issue` | `(self, project_id: Optional[str] = None, issue_iid: Optional[int] = None, title: Optional[str] = None, description: Optional[str] = None, labels: Optional[list] = None, **kwargs) -> ToolResult` | 411 | Update an issue. |
| ⚙️ | `_close_issue` | `(self, project_id: Optional[str] = None, issue_iid: Optional[int] = None, **kwargs) -> ToolResult` | 446 | Close an issue. |
| ⚙️ | `_list_merge_requests` | `(self, project_id: Optional[str] = None, state: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 472 | List merge requests. |
| ⚙️ | `_get_merge_request` | `(self, project_id: Optional[str] = None, mr_iid: Optional[int] = None, **kwargs) -> ToolResult` | 510 | Get merge request details. |
| ⚙️ | `_create_merge_request` | `(self, project_id: Optional[str] = None, source_branch: Optional[str] = None, target_branch: Optional[str] = None, title: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 539 | Create a merge request. |
| ⚙️ | `_merge_merge_request` | `(self, project_id: Optional[str] = None, mr_iid: Optional[int] = None, **kwargs) -> ToolResult` | 577 | Merge a merge request. |
| ⚙️ | `_close_merge_request` | `(self, project_id: Optional[str] = None, mr_iid: Optional[int] = None, **kwargs) -> ToolResult` | 608 | Close a merge request. |
| ⚙️ | `_list_pipelines` | `(self, project_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 634 | List pipelines. |
| ⚙️ | `_get_pipeline` | `(self, project_id: Optional[str] = None, pipeline_id: Optional[int] = None, **kwargs) -> ToolResult` | 668 | Get pipeline details. |
| ⚙️ | `_create_pipeline` | `(self, project_id: Optional[str] = None, branch: Optional[str] = None, **kwargs) -> ToolResult` | 695 | Create a pipeline. |
| ⚙️ | `_cancel_pipeline` | `(self, project_id: Optional[str] = None, pipeline_id: Optional[int] = None, **kwargs) -> ToolResult` | 721 | Cancel a pipeline. |
| ⚙️ | `_retry_pipeline` | `(self, project_id: Optional[str] = None, pipeline_id: Optional[int] = None, **kwargs) -> ToolResult` | 742 | Retry a pipeline. |
| ⚙️ | `_list_jobs` | `(self, project_id: Optional[str] = None, pipeline_id: Optional[int] = None, limit: int = 100, **kwargs) -> ToolResult` | 764 | List jobs. |
| ⚙️ | `_get_job` | `(self, project_id: Optional[str] = None, job_id: Optional[int] = None, **kwargs) -> ToolResult` | 801 | Get job details. |
| ⚙️ | `_retry_job` | `(self, project_id: Optional[str] = None, job_id: Optional[int] = None, **kwargs) -> ToolResult` | 829 | Retry a job. |
| ⚙️ | `_cancel_job` | `(self, project_id: Optional[str] = None, job_id: Optional[int] = None, **kwargs) -> ToolResult` | 850 | Cancel a job. |
| ⚙️ | `_list_branches` | `(self, project_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 872 | List branches. |
| ⚙️ | `_get_branch` | `(self, project_id: Optional[str] = None, branch: Optional[str] = None, **kwargs) -> ToolResult` | 905 | Get branch details. |
| ⚙️ | `_create_branch` | `(self, project_id: Optional[str] = None, branch: Optional[str] = None, **kwargs) -> ToolResult` | 930 | Create a branch. |
| ⚙️ | `_delete_branch` | `(self, project_id: Optional[str] = None, branch: Optional[str] = None, **kwargs) -> ToolResult` | 956 | Delete a branch. |
| ⚙️ | `_search` | `(self, query: Optional[str] = None, **kwargs) -> ToolResult` | 978 | Search across GitLab. |
| ⚙️ | `_me` | `(self, **kwargs) -> ToolResult` | 1003 | Get current user. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 1017 |  |

---

## `argus/tools/integrations/devops/n8n.py`

### 🏛️ `class N8nTool(BaseTool)`  <sub>line 18</sub>

> n8n - Workflow automation platform.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, base_url: Optional[str] = None, config: Optional[ToolConfig] = None)` | 40 |  |
| ⚙️ | `_get_session` | `(self)` | 58 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict \| list` | 69 | Make API request to n8n. |
| ⚙️ | `execute` | `(self, action: str = 'list_workflows', workflow_id: Optional[str] = None, execution_id: Optional[str] = None, credential_id: Optional[str] = None, tag_id: Optional[str] = None, name: Optional[str] = None, active: Optional[bool] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 99 | Execute n8n operations. |
| ⚙️ | `_list_workflows` | `(self, active: Optional[bool] = None, limit: int = 100, **kwargs) -> ToolResult` | 169 | List workflows. |
| ⚙️ | `_get_workflow` | `(self, workflow_id: Optional[str] = None, **kwargs) -> ToolResult` | 205 | Get workflow details. |
| ⚙️ | `_create_workflow` | `(self, name: Optional[str] = None, **kwargs) -> ToolResult` | 229 | Create a workflow. |
| ⚙️ | `_update_workflow` | `(self, workflow_id: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 266 | Update a workflow. |
| ⚙️ | `_delete_workflow` | `(self, workflow_id: Optional[str] = None, **kwargs) -> ToolResult` | 293 | Delete a workflow. |
| ⚙️ | `_activate_workflow` | `(self, workflow_id: Optional[str] = None, **kwargs) -> ToolResult` | 309 | Activate a workflow. |
| ⚙️ | `_deactivate_workflow` | `(self, workflow_id: Optional[str] = None, **kwargs) -> ToolResult` | 326 | Deactivate a workflow. |
| ⚙️ | `_execute_workflow` | `(self, workflow_id: Optional[str] = None, **kwargs) -> ToolResult` | 343 | Execute a workflow. |
| ⚙️ | `_list_executions` | `(self, workflow_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 371 | List executions. |
| ⚙️ | `_get_execution` | `(self, execution_id: Optional[str] = None, **kwargs) -> ToolResult` | 408 | Get execution details. |
| ⚙️ | `_delete_execution` | `(self, execution_id: Optional[str] = None, **kwargs) -> ToolResult` | 435 | Delete an execution. |
| ⚙️ | `_list_credentials` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 452 | List credentials. |
| ⚙️ | `_get_credential` | `(self, credential_id: Optional[str] = None, **kwargs) -> ToolResult` | 479 | Get credential details (without sensitive data). |
| ⚙️ | `_create_credential` | `(self, name: Optional[str] = None, **kwargs) -> ToolResult` | 500 | Create a credential. |
| ⚙️ | `_delete_credential` | `(self, credential_id: Optional[str] = None, **kwargs) -> ToolResult` | 529 | Delete a credential. |
| ⚙️ | `_list_tags` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 546 | List tags. |
| ⚙️ | `_get_tag` | `(self, tag_id: Optional[str] = None, **kwargs) -> ToolResult` | 572 | Get tag details. |
| ⚙️ | `_create_tag` | `(self, name: Optional[str] = None, **kwargs) -> ToolResult` | 592 | Create a tag. |
| ⚙️ | `_update_tag` | `(self, tag_id: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 609 | Update a tag. |
| ⚙️ | `_delete_tag` | `(self, tag_id: Optional[str] = None, **kwargs) -> ToolResult` | 629 | Delete a tag. |
| ⚙️ | `_test_webhook` | `(self, workflow_id: Optional[str] = None, **kwargs) -> ToolResult` | 646 | Test a webhook trigger. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 674 |  |

---

## `argus/tools/integrations/devops/postman.py`

### 🏛️ `class PostmanTool(BaseTool)`  <sub>line 18</sub>

> Postman - API testing and documentation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 42 |  |
| ⚙️ | `_get_session` | `(self)` | 58 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 69 | Make API request to Postman. |
| ⚙️ | `execute` | `(self, action: str = 'list_collections', collection_id: Optional[str] = None, environment_id: Optional[str] = None, workspace_id: Optional[str] = None, monitor_id: Optional[str] = None, mock_id: Optional[str] = None, api_id: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None, schema: Optional[dict] = None, **kwargs: Any) -> ToolResult` | 98 | Execute Postman operations. |
| ⚙️ | `_list_collections` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 179 | List collections. |
| ⚙️ | `_get_collection` | `(self, collection_id: Optional[str] = None, **kwargs) -> ToolResult` | 207 | Get collection details. |
| ⚙️ | `_create_collection` | `(self, name: Optional[str] = None, description: Optional[str] = None, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 232 | Create a collection. |
| ⚙️ | `_update_collection` | `(self, collection_id: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 276 | Update a collection. |
| ⚙️ | `_delete_collection` | `(self, collection_id: Optional[str] = None, **kwargs) -> ToolResult` | 307 | Delete a collection. |
| ⚙️ | `_fork_collection` | `(self, collection_id: Optional[str] = None, workspace_id: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 323 | Fork a collection. |
| ⚙️ | `_list_environments` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 351 | List environments. |
| ⚙️ | `_get_environment` | `(self, environment_id: Optional[str] = None, **kwargs) -> ToolResult` | 378 | Get environment details. |
| ⚙️ | `_create_environment` | `(self, name: Optional[str] = None, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 399 | Create an environment. |
| ⚙️ | `_update_environment` | `(self, environment_id: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 438 | Update an environment. |
| ⚙️ | `_delete_environment` | `(self, environment_id: Optional[str] = None, **kwargs) -> ToolResult` | 471 | Delete an environment. |
| ⚙️ | `_list_workspaces` | `(self, **kwargs) -> ToolResult` | 488 | List workspaces. |
| ⚙️ | `_get_workspace` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 513 | Get workspace details. |
| ⚙️ | `_create_workspace` | `(self, name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 539 | Create a workspace. |
| ⚙️ | `_list_monitors` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 572 | List monitors. |
| ⚙️ | `_get_monitor` | `(self, monitor_id: Optional[str] = None, **kwargs) -> ToolResult` | 599 | Get monitor details. |
| ⚙️ | `_run_monitor` | `(self, monitor_id: Optional[str] = None, **kwargs) -> ToolResult` | 623 | Run a monitor. |
| ⚙️ | `_list_mocks` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 644 | List mock servers. |
| ⚙️ | `_get_mock` | `(self, mock_id: Optional[str] = None, **kwargs) -> ToolResult` | 672 | Get mock server details. |
| ⚙️ | `_create_mock` | `(self, collection_id: Optional[str] = None, name: Optional[str] = None, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 696 | Create a mock server. |
| ⚙️ | `_list_apis` | `(self, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 740 | List APIs. |
| ⚙️ | `_get_api` | `(self, api_id: Optional[str] = None, **kwargs) -> ToolResult` | 767 | Get API details. |
| ⚙️ | `_create_api` | `(self, name: Optional[str] = None, description: Optional[str] = None, workspace_id: Optional[str] = None, **kwargs) -> ToolResult` | 791 | Create an API. |
| ⚙️ | `_me` | `(self, **kwargs) -> ToolResult` | 831 | Get current user. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 846 |  |

---

## `argus/tools/integrations/finance/weather.py`

### 🏛️ `class WeatherTool(BaseTool)`  <sub>line 18</sub>

> OpenWeatherMap weather tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 33 |  |
| ⚙️ | `execute` | `(self, location: str = '', action: str = 'current', units: str = 'metric', **kwargs: Any) -> ToolResult` | 41 | Get weather data. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 120 |  |

---

## `argus/tools/integrations/finance/yahoo_finance.py`

### 🏛️ `class YahooFinanceTool(BaseTool)`  <sub>line 17</sub>

> Yahoo Finance data tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ToolConfig] = None)` | 30 |  |
| ⚙️ | `execute` | `(self, symbol: str = '', action: str = 'quote', period: str = '1mo', **kwargs: Any) -> ToolResult` | 33 | Get financial data. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 128 |  |

---

## `argus/tools/integrations/media_ai/cartesia.py`

### 🏛️ `class CartesiaTool(BaseTool)`  <sub>line 19</sub>

> Cartesia - Real-time voice AI platform.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 43 |  |
| ⚙️ | `_get_session` | `(self)` | 59 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None, accept_audio: bool = False) -> dict \| bytes` | 71 | Make API request to Cartesia. |
| ⚙️ | `execute` | `(self, action: str = 'list_voices', voice_id: Optional[str] = None, text: Optional[str] = None, model_id: Optional[str] = None, name: Optional[str] = None, language: Optional[str] = None, **kwargs: Any) -> ToolResult` | 110 | Execute Cartesia operations. |
| ⚙️ | `_list_voices` | `(self, **kwargs) -> ToolResult` | 163 | List available voices. |
| ⚙️ | `_get_voice` | `(self, voice_id: Optional[str] = None, **kwargs) -> ToolResult` | 183 | Get voice details. |
| ⚙️ | `_create_voice` | `(self, name: Optional[str] = None, language: Optional[str] = None, **kwargs) -> ToolResult` | 206 | Create a voice from embedding. |
| ⚙️ | `_clone_voice` | `(self, name: Optional[str] = None, **kwargs) -> ToolResult` | 240 | Clone a voice from audio samples. |
| ⚙️ | `_update_voice` | `(self, voice_id: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 292 | Update a voice. |
| ⚙️ | `_delete_voice` | `(self, voice_id: Optional[str] = None, **kwargs) -> ToolResult` | 321 | Delete a voice. |
| ⚙️ | `_text_to_speech` | `(self, voice_id: Optional[str] = None, text: Optional[str] = None, model_id: Optional[str] = None, language: Optional[str] = None, **kwargs) -> ToolResult` | 338 | Generate speech from text. |
| ⚙️ | `_text_to_speech_bytes` | `(self, voice_id: Optional[str] = None, text: Optional[str] = None, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 395 | Generate speech and return raw bytes info. |
| ⚙️ | `_create_embedding` | `(self, **kwargs) -> ToolResult` | 411 | Create voice embedding from audio. |
| ⚙️ | `_list_languages` | `(self, **kwargs) -> ToolResult` | 436 | List supported languages. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 462 |  |

---

## `argus/tools/integrations/media_ai/elevenlabs.py`

### 🏛️ `class ElevenLabsTool(BaseTool)`  <sub>line 19</sub>

> ElevenLabs - AI voice synthesis platform.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 44 |  |
| ⚙️ | `_get_session` | `(self)` | 60 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None, accept_audio: bool = False) -> dict \| bytes` | 71 | Make API request to ElevenLabs. |
| ⚙️ | `execute` | `(self, action: str = 'list_voices', voice_id: Optional[str] = None, text: Optional[str] = None, model_id: Optional[str] = None, name: Optional[str] = None, history_item_id: Optional[str] = None, **kwargs: Any) -> ToolResult` | 112 | Execute ElevenLabs operations. |
| ⚙️ | `_list_voices` | `(self, **kwargs) -> ToolResult` | 175 | List available voices. |
| ⚙️ | `_get_voice` | `(self, voice_id: Optional[str] = None, **kwargs) -> ToolResult` | 194 | Get voice details. |
| ⚙️ | `_add_voice` | `(self, name: Optional[str] = None, **kwargs) -> ToolResult` | 226 | Add a new voice (voice cloning). |
| ⚙️ | `_edit_voice` | `(self, voice_id: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 284 | Edit a voice. |
| ⚙️ | `_delete_voice` | `(self, voice_id: Optional[str] = None, **kwargs) -> ToolResult` | 323 | Delete a voice. |
| ⚙️ | `_get_voice_settings` | `(self, voice_id: Optional[str] = None, **kwargs) -> ToolResult` | 339 | Get voice settings. |
| ⚙️ | `_text_to_speech` | `(self, voice_id: Optional[str] = None, text: Optional[str] = None, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 360 | Generate speech from text. |
| ⚙️ | `_text_to_speech_stream` | `(self, voice_id: Optional[str] = None, text: Optional[str] = None, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 408 | Generate speech with streaming (returns audio chunks). |
| ⚙️ | `_list_models` | `(self, **kwargs) -> ToolResult` | 425 | List available models. |
| ⚙️ | `_list_history` | `(self, **kwargs) -> ToolResult` | 447 | List generation history. |
| ⚙️ | `_get_history_item` | `(self, history_item_id: Optional[str] = None, **kwargs) -> ToolResult` | 485 | Get history item details. |
| ⚙️ | `_delete_history_item` | `(self, history_item_id: Optional[str] = None, **kwargs) -> ToolResult` | 507 | Delete a history item. |
| ⚙️ | `_download_history_audio` | `(self, history_item_id: Optional[str] = None, **kwargs) -> ToolResult` | 523 | Download audio from history. |
| ⚙️ | `_generate_sound_effect` | `(self, text: Optional[str] = None, **kwargs) -> ToolResult` | 547 | Generate sound effects from text description. |
| ⚙️ | `_get_user` | `(self, **kwargs) -> ToolResult` | 583 | Get current user info. |
| ⚙️ | `_get_subscription` | `(self, **kwargs) -> ToolResult` | 596 | Get subscription info. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 613 |  |

---

## `argus/tools/integrations/media_ai/huggingface.py`

### 🏛️ `class HuggingFaceTool(BaseTool)`  <sub>line 19</sub>

> Hugging Face - ML model hub and inference API.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_token: Optional[str] = None, config: Optional[ToolConfig] = None)` | 44 |  |
| ⚙️ | `_get_session` | `(self)` | 60 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, url: str, data: Optional[dict] = None, params: Optional[dict] = None, raw_input: bool = False) -> dict \| list \| bytes` | 71 | Make API request. |
| ⚙️ | `execute` | `(self, action: str = 'search_models', model: Optional[str] = None, dataset: Optional[str] = None, space: Optional[str] = None, query: Optional[str] = None, inputs: Optional[Any] = None, task: Optional[str] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 110 | Execute Hugging Face operations. |
| ⚙️ | `_search_models` | `(self, query: Optional[str] = None, task: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 182 | Search for models. |
| ⚙️ | `_get_model` | `(self, model: Optional[str] = None, **kwargs) -> ToolResult` | 224 | Get model details. |
| ⚙️ | `_list_model_files` | `(self, model: Optional[str] = None, **kwargs) -> ToolResult` | 249 | List model files. |
| ⚙️ | `_search_datasets` | `(self, query: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 275 | Search for datasets. |
| ⚙️ | `_get_dataset` | `(self, dataset: Optional[str] = None, **kwargs) -> ToolResult` | 310 | Get dataset details. |
| ⚙️ | `_search_spaces` | `(self, query: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 333 | Search for spaces. |
| ⚙️ | `_get_space` | `(self, space: Optional[str] = None, **kwargs) -> ToolResult` | 368 | Get space details. |
| ⚙️ | `_inference` | `(self, model: Optional[str] = None, inputs: Optional[Any] = None, **kwargs) -> ToolResult` | 391 | Run inference on a model. |
| ⚙️ | `_text_generation` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 428 | Generate text. |
| ⚙️ | `_text_classification` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 454 | Classify text. |
| ⚙️ | `_token_classification` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 464 | Token classification (NER, POS tagging). |
| ⚙️ | `_question_answering` | `(self, model: Optional[str] = None, inputs: Optional[dict] = None, **kwargs) -> ToolResult` | 474 | Answer questions. |
| ⚙️ | `_summarization` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 492 | Summarize text. |
| ⚙️ | `_translation` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 508 | Translate text. |
| ⚙️ | `_fill_mask` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 518 | Fill masked tokens. |
| ⚙️ | `_feature_extraction` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 528 | Extract features/embeddings. |
| ⚙️ | `_image_classification` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 538 | Classify images. |
| ⚙️ | `_object_detection` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 570 | Detect objects in images. |
| ⚙️ | `_image_to_text` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 580 | Generate text from images (captioning, OCR). |
| ⚙️ | `_text_to_image` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 590 | Generate images from text. |
| ⚙️ | `_text_to_speech` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 614 | Generate speech from text. |
| ⚙️ | `_asr` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 638 | Automatic speech recognition. |
| ⚙️ | `_zero_shot_classification` | `(self, model: Optional[str] = None, inputs: Optional[str] = None, **kwargs) -> ToolResult` | 668 | Zero-shot classification. |
| ⚙️ | `_sentence_similarity` | `(self, model: Optional[str] = None, inputs: Optional[dict] = None, **kwargs) -> ToolResult` | 691 | Compute sentence similarity. |
| ⚙️ | `_conversational` | `(self, model: Optional[str] = None, inputs: Optional[dict] = None, **kwargs) -> ToolResult` | 709 | Conversational response. |
| ⚙️ | `_whoami` | `(self, **kwargs) -> ToolResult` | 735 | Get current user info. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 752 |  |

---

## `argus/tools/integrations/observability/arize.py`

### 🏛️ `class ArizeTool(BaseTool)`  <sub>line 19</sub>

> Arize - ML observability and model monitoring.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, space_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 43 |  |
| ⚙️ | `_get_session` | `(self)` | 61 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 72 | Make API request to Arize. |
| ⚙️ | `execute` | `(self, action: str = 'list_models', model_id: Optional[str] = None, model_version: Optional[str] = None, prediction_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs: Any) -> ToolResult` | 100 | Execute Arize operations. |
| ⚙️ | `_list_models` | `(self, **kwargs) -> ToolResult` | 166 | List models. |
| ⚙️ | `_get_model` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 181 | Get model details. |
| ⚙️ | `_get_model_metrics` | `(self, model_id: Optional[str] = None, model_version: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs) -> ToolResult` | 204 | Get model metrics. |
| ⚙️ | `_get_model_performance` | `(self, model_id: Optional[str] = None, model_version: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs) -> ToolResult` | 237 | Get model performance metrics. |
| ⚙️ | `_log_prediction` | `(self, model_id: Optional[str] = None, model_version: Optional[str] = None, prediction_id: Optional[str] = None, **kwargs) -> ToolResult` | 275 | Log a prediction. |
| ⚙️ | `_log_actual` | `(self, model_id: Optional[str] = None, prediction_id: Optional[str] = None, **kwargs) -> ToolResult` | 323 | Log actual/ground truth value. |
| ⚙️ | `_log_batch` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 362 | Log batch of predictions. |
| ⚙️ | `_get_drift_metrics` | `(self, model_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs) -> ToolResult` | 392 | Get drift metrics. |
| ⚙️ | `_get_feature_drift` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 422 | Get per-feature drift metrics. |
| ⚙️ | `_get_data_quality` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 451 | Get data quality metrics. |
| ⚙️ | `_get_missing_values` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 475 | Get missing value statistics. |
| ⚙️ | `_list_alerts` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 500 | List alerts. |
| ⚙️ | `_create_alert` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 519 | Create an alert. |
| ⚙️ | `_delete_alert` | `(self, **kwargs) -> ToolResult` | 556 | Delete an alert. |
| ⚙️ | `_list_monitors` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 577 | List monitors. |
| ⚙️ | `_get_monitor` | `(self, **kwargs) -> ToolResult` | 596 | Get monitor details. |
| ⚙️ | `_get_feature_importance` | `(self, model_id: Optional[str] = None, **kwargs) -> ToolResult` | 620 | Get feature importance. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 644 |  |

---

## `argus/tools/integrations/observability/mlflow_tool.py`

### 🏛️ `class MLflowTool(BaseTool)`  <sub>line 18</sub>

> MLflow - ML lifecycle management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, tracking_uri: Optional[str] = None, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_session` | `(self)` | 56 | Get HTTP session. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict \| list` | 67 | Make API request to MLflow. |
| ⚙️ | `execute` | `(self, action: str = 'list_experiments', experiment_id: Optional[str] = None, run_id: Optional[str] = None, model_name: Optional[str] = None, version: Optional[str] = None, **kwargs: Any) -> ToolResult` | 97 | Execute MLflow operations. |
| ⚙️ | `_list_experiments` | `(self, **kwargs) -> ToolResult` | 174 | List experiments. |
| ⚙️ | `_get_experiment` | `(self, experiment_id: Optional[str] = None, **kwargs) -> ToolResult` | 195 | Get experiment details. |
| ⚙️ | `_create_experiment` | `(self, **kwargs) -> ToolResult` | 222 | Create an experiment. |
| ⚙️ | `_delete_experiment` | `(self, experiment_id: Optional[str] = None, **kwargs) -> ToolResult` | 248 | Delete an experiment. |
| ⚙️ | `_update_experiment` | `(self, experiment_id: Optional[str] = None, **kwargs) -> ToolResult` | 264 | Update an experiment. |
| ⚙️ | `_search_experiments` | `(self, **kwargs) -> ToolResult` | 286 | Search experiments. |
| ⚙️ | `_list_runs` | `(self, experiment_id: Optional[str] = None, **kwargs) -> ToolResult` | 317 | List runs. |
| ⚙️ | `_get_run` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 346 | Get run details. |
| ⚙️ | `_create_run` | `(self, experiment_id: Optional[str] = None, **kwargs) -> ToolResult` | 361 | Create a run. |
| ⚙️ | `_update_run` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 394 | Update a run. |
| ⚙️ | `_delete_run` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 424 | Delete a run. |
| ⚙️ | `_search_runs` | `(self, experiment_id: Optional[str] = None, **kwargs) -> ToolResult` | 440 | Search runs. |
| ⚙️ | `_log_metric` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 482 | Log a metric. |
| ⚙️ | `_log_metrics` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 519 | Log multiple metrics. |
| ⚙️ | `_log_param` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 555 | Log a parameter. |
| ⚙️ | `_log_params` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 586 | Log multiple parameters. |
| ⚙️ | `_get_metric_history` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 614 | Get metric history. |
| ⚙️ | `_set_tag` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 639 | Set a tag on a run. |
| ⚙️ | `_delete_tag` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 670 | Delete a tag from a run. |
| ⚙️ | `_list_registered_models` | `(self, **kwargs) -> ToolResult` | 697 | List registered models. |
| ⚙️ | `_get_registered_model` | `(self, model_name: Optional[str] = None, **kwargs) -> ToolResult` | 716 | Get registered model details. |
| ⚙️ | `_create_registered_model` | `(self, **kwargs) -> ToolResult` | 735 | Create a registered model. |
| ⚙️ | `_delete_registered_model` | `(self, model_name: Optional[str] = None, **kwargs) -> ToolResult` | 761 | Delete a registered model. |
| ⚙️ | `_update_registered_model` | `(self, model_name: Optional[str] = None, **kwargs) -> ToolResult` | 781 | Update a registered model. |
| ⚙️ | `_search_registered_models` | `(self, **kwargs) -> ToolResult` | 803 | Search registered models. |
| ⚙️ | `_list_model_versions` | `(self, model_name: Optional[str] = None, **kwargs) -> ToolResult` | 831 | List model versions. |
| ⚙️ | `_get_model_version` | `(self, model_name: Optional[str] = None, version: Optional[str] = None, **kwargs) -> ToolResult` | 858 | Get model version details. |
| ⚙️ | `_create_model_version` | `(self, model_name: Optional[str] = None, **kwargs) -> ToolResult` | 880 | Create a model version. |
| ⚙️ | `_update_model_version` | `(self, model_name: Optional[str] = None, version: Optional[str] = None, **kwargs) -> ToolResult` | 917 | Update a model version. |
| ⚙️ | `_delete_model_version` | `(self, model_name: Optional[str] = None, version: Optional[str] = None, **kwargs) -> ToolResult` | 946 | Delete a model version. |
| ⚙️ | `_transition_model_stage` | `(self, model_name: Optional[str] = None, version: Optional[str] = None, **kwargs) -> ToolResult` | 970 | Transition model version to a stage. |
| ⚙️ | `_list_artifacts` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 1003 | List artifacts for a run. |
| ⚙️ | `_log_artifact` | `(self, run_id: Optional[str] = None, **kwargs) -> ToolResult` | 1025 | Log an artifact for a run. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 1069 |  |

---

## `argus/tools/integrations/observability/monocle.py`

### 🏛️ `class MonocleTool(BaseTool)`  <sub>line 19</sub>

> Monocle - GenAI application observability.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, base_url: Optional[str] = None, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_session` | `(self)` | 57 | Get HTTP session. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict \| list` | 68 | Make API request to Monocle. |
| ⚙️ | `execute` | `(self, action: str = 'list_traces', trace_id: Optional[str] = None, span_id: Optional[str] = None, workflow_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 98 | Execute Monocle operations. |
| ⚙️ | `_list_traces` | `(self, workflow_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 161 | List traces. |
| ⚙️ | `_get_trace` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 188 | Get trace details. |
| ⚙️ | `_log_trace` | `(self, **kwargs) -> ToolResult` | 203 | Log a trace. |
| ⚙️ | `_list_spans` | `(self, trace_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 239 | List spans. |
| ⚙️ | `_get_span` | `(self, span_id: Optional[str] = None, **kwargs) -> ToolResult` | 264 | Get span details. |
| ⚙️ | `_log_span` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 279 | Log a span. |
| ⚙️ | `_list_workflows` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 321 | List workflows. |
| ⚙️ | `_get_workflow` | `(self, workflow_name: Optional[str] = None, **kwargs) -> ToolResult` | 338 | Get workflow details. |
| ⚙️ | `_log_workflow` | `(self, workflow_name: Optional[str] = None, **kwargs) -> ToolResult` | 353 | Log a workflow execution. |
| ⚙️ | `_log_llm_call` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 391 | Log an LLM call. |
| ⚙️ | `_log_retrieval` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 447 | Log a retrieval operation. |
| ⚙️ | `_log_embedding` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 490 | Log an embedding operation. |
| ⚙️ | `_get_latency_stats` | `(self, workflow_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs) -> ToolResult` | 531 | Get latency statistics. |
| ⚙️ | `_get_token_usage` | `(self, workflow_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs) -> ToolResult` | 558 | Get token usage statistics. |
| ⚙️ | `_get_cost_analysis` | `(self, workflow_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs) -> ToolResult` | 585 | Get cost analysis. |
| ⚙️ | `_setup_tracing` | `(self, **kwargs) -> ToolResult` | 613 | Setup tracing configuration. |
| ⚙️ | `_get_config` | `(self, **kwargs) -> ToolResult` | 637 | Get current configuration. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 648 |  |

---

## `argus/tools/integrations/observability/phoenix.py`

### 🏛️ `class PhoenixTool(BaseTool)`  <sub>line 19</sub>

> Arize Phoenix - LLM observability and evaluation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, base_url: Optional[str] = None, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_session` | `(self)` | 56 | Get HTTP session. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict \| list` | 67 | Make API request to Phoenix. |
| ⚙️ | `execute` | `(self, action: str = 'list_traces', trace_id: Optional[str] = None, span_id: Optional[str] = None, project_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 97 | Execute Phoenix operations. |
| ⚙️ | `_list_traces` | `(self, project_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 163 | List traces. |
| ⚙️ | `_get_trace` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 190 | Get trace details. |
| ⚙️ | `_log_trace` | `(self, **kwargs) -> ToolResult` | 205 | Log a trace. |
| ⚙️ | `_delete_trace` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 240 | Delete a trace. |
| ⚙️ | `_list_spans` | `(self, trace_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 257 | List spans. |
| ⚙️ | `_get_span` | `(self, span_id: Optional[str] = None, **kwargs) -> ToolResult` | 278 | Get span details. |
| ⚙️ | `_log_span` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 293 | Log a span. |
| ⚙️ | `_list_evaluations` | `(self, project_name: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 341 | List evaluations. |
| ⚙️ | `_log_evaluation` | `(self, span_id: Optional[str] = None, **kwargs) -> ToolResult` | 362 | Log an evaluation. |
| ⚙️ | `_get_evaluation_metrics` | `(self, project_name: Optional[str] = None, **kwargs) -> ToolResult` | 399 | Get evaluation metrics summary. |
| ⚙️ | `_list_datasets` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 420 | List datasets. |
| ⚙️ | `_get_dataset` | `(self, **kwargs) -> ToolResult` | 437 | Get dataset details. |
| ⚙️ | `_create_dataset` | `(self, **kwargs) -> ToolResult` | 452 | Create a dataset. |
| ⚙️ | `_list_projects` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 479 | List projects. |
| ⚙️ | `_get_project` | `(self, project_name: Optional[str] = None, **kwargs) -> ToolResult` | 496 | Get project details. |
| ⚙️ | `_get_embedding_drift` | `(self, project_name: Optional[str] = None, **kwargs) -> ToolResult` | 512 | Get embedding drift metrics. |
| ⚙️ | `_get_clusters` | `(self, project_name: Optional[str] = None, **kwargs) -> ToolResult` | 532 | Get embedding clusters. |
| ⚙️ | `_export_traces` | `(self, project_name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, **kwargs) -> ToolResult` | 557 | Export traces. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 587 |  |

---

## `argus/tools/integrations/observability/wandb_weave.py`

### 🏛️ `class WandBWeaveTool(BaseTool)`  <sub>line 19</sub>

> Weights & Biases Weave - LLM observability and evaluation.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, entity: Optional[str] = None, project: Optional[str] = None, config: Optional[ToolConfig] = None)` | 42 |  |
| ⚙️ | `_get_session` | `(self)` | 61 | Get HTTP session. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None, base_url: Optional[str] = None) -> dict \| list` | 74 | Make API request to W&B/Weave. |
| ⚙️ | `_graphql` | `(self, query: str, variables: Optional[dict] = None) -> dict` | 105 | Execute GraphQL query against W&B API. |
| ⚙️ | `execute` | `(self, action: str = 'list_calls', call_id: Optional[str] = None, trace_id: Optional[str] = None, evaluation_id: Optional[str] = None, dataset_id: Optional[str] = None, **kwargs: Any) -> ToolResult` | 130 | Execute W&B Weave operations. |
| ⚙️ | `_list_calls` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 197 | List calls/spans. |
| ⚙️ | `_get_call` | `(self, call_id: Optional[str] = None, **kwargs) -> ToolResult` | 239 | Get call details. |
| ⚙️ | `_log_call` | `(self, **kwargs) -> ToolResult` | 257 | Log a call/span. |
| ⚙️ | `_list_traces` | `(self, **kwargs) -> ToolResult` | 318 | List traces. |
| ⚙️ | `_get_trace` | `(self, trace_id: Optional[str] = None, **kwargs) -> ToolResult` | 352 | Get trace details. |
| ⚙️ | `_list_evaluations` | `(self, **kwargs) -> ToolResult` | 371 | List evaluations. |
| ⚙️ | `_get_evaluation` | `(self, evaluation_id: Optional[str] = None, **kwargs) -> ToolResult` | 397 | Get evaluation details. |
| ⚙️ | `_create_evaluation` | `(self, **kwargs) -> ToolResult` | 415 | Create an evaluation. |
| ⚙️ | `_log_evaluation_result` | `(self, evaluation_id: Optional[str] = None, **kwargs) -> ToolResult` | 455 | Log an evaluation result. |
| ⚙️ | `_list_datasets` | `(self, **kwargs) -> ToolResult` | 496 | List datasets. |
| ⚙️ | `_get_dataset` | `(self, dataset_id: Optional[str] = None, **kwargs) -> ToolResult` | 522 | Get dataset details. |
| ⚙️ | `_create_dataset` | `(self, **kwargs) -> ToolResult` | 540 | Create a dataset. |
| ⚙️ | `_add_dataset_rows` | `(self, dataset_id: Optional[str] = None, **kwargs) -> ToolResult` | 572 | Add rows to a dataset. |
| ⚙️ | `_list_models` | `(self, **kwargs) -> ToolResult` | 599 | List models. |
| ⚙️ | `_get_model` | `(self, **kwargs) -> ToolResult` | 625 | Get model details. |
| ⚙️ | `_log_model` | `(self, **kwargs) -> ToolResult` | 643 | Log a model. |
| ⚙️ | `_list_ops` | `(self, **kwargs) -> ToolResult` | 680 | List ops (traced functions). |
| ⚙️ | `_get_op` | `(self, **kwargs) -> ToolResult` | 706 | Get op (traced function) details. |
| ⚙️ | `_get_call_stats` | `(self, **kwargs) -> ToolResult` | 727 | Get call statistics. |
| ⚙️ | `_get_cost_summary` | `(self, **kwargs) -> ToolResult` | 759 | Get cost summary. |
| ⚙️ | `_get_latency_stats` | `(self, **kwargs) -> ToolResult` | 790 | Get latency statistics. |
| ⚙️ | `_list_projects` | `(self, **kwargs) -> ToolResult` | 826 | List projects. |
| ⚙️ | `_get_project` | `(self, **kwargs) -> ToolResult` | 863 | Get project details. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 892 |  |

---

## `argus/tools/integrations/productivity/filesystem.py`

### 🏛️ `class FileSystemTool(BaseTool)`  <sub>line 19</sub>

> File system operations tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, base_dir: Optional[str] = None, allow_write: bool = True, config: Optional[ToolConfig] = None)` | 32 |  |
| ⚙️ | `_safe_path` | `(self, path: str) -> Path` | 42 | Ensure path is within base directory. |
| ⚙️ | `execute` | `(self, action: str = 'list', path: str = '.', content: Optional[str] = None, **kwargs: Any) -> ToolResult` | 49 | Perform file system operation. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 119 |  |

---

## `argus/tools/integrations/productivity/github.py`

### 🏛️ `class GitHubTool(BaseTool)`  <sub>line 18</sub>

> GitHub API tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, token: Optional[str] = None, config: Optional[ToolConfig] = None)` | 31 |  |
| ⚙️ | `execute` | `(self, action: str = 'search_repos', query: Optional[str] = None, owner: Optional[str] = None, repo: Optional[str] = None, path: Optional[str] = None, **kwargs: Any) -> ToolResult` | 39 | Perform GitHub operation. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 139 |  |

---

## `argus/tools/integrations/productivity/json_tool.py`

### 🏛️ `class JsonTool(BaseTool)`  <sub>line 18</sub>

> JSON manipulation tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ToolConfig] = None)` | 31 |  |
| ⚙️ | `execute` | `(self, action: str = 'parse', data: str = '', path: Optional[str] = None, value: Optional[Any] = None, **kwargs: Any) -> ToolResult` | 34 | Perform JSON operation. |
| ⚙️ | `_get_path` | `(self, data: Any, path: str) -> Any` | 97 | Get value at path. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 108 |  |

---

## `argus/tools/integrations/productivity/python_repl.py`

### 🏛️ `class PythonReplTool(BaseTool)`  <sub>line 19</sub>

> Python code execution tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, globals_dict: Optional[dict] = None, timeout: float = 30.0, config: Optional[ToolConfig] = None)` | 32 |  |
| ⚙️ | `execute` | `(self, code: str = '', reset: bool = False, **kwargs: Any) -> ToolResult` | 43 | Execute Python code. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 102 |  |

---

## `argus/tools/integrations/productivity/shell.py`

### 🏛️ `class ShellTool(BaseTool)`  <sub>line 18</sub>

> Shell command execution tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, allowed_commands: Optional[list[str]] = None, timeout: float = 30.0, cwd: Optional[str] = None, config: Optional[ToolConfig] = None)` | 34 |  |
| ⚙️ | `_is_safe` | `(self, command: str) -> bool` | 46 | Check if command is safe to execute. |
| ⚙️ | `execute` | `(self, command: str = '', timeout: Optional[float] = None, **kwargs: Any) -> ToolResult` | 53 | Execute shell command. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 105 |  |

---

## `argus/tools/integrations/productivity_tools/asana.py`

### 🏛️ `class AsanaTool(BaseTool)`  <sub>line 19</sub>

> Asana - Project management and task tracking.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, access_token: Optional[str] = None, workspace_gid: Optional[str] = None, config: Optional[ToolConfig] = None)` | 45 |  |
| ⚙️ | `_get_session` | `(self)` | 63 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 75 | Make API request to Asana. |
| ⚙️ | `execute` | `(self, action: str = 'list_workspaces', workspace_gid: Optional[str] = None, project_gid: Optional[str] = None, task_gid: Optional[str] = None, section_gid: Optional[str] = None, name: Optional[str] = None, notes: Optional[str] = None, due_on: Optional[str] = None, assignee: Optional[str] = None, completed: Optional[bool] = None, tags: Optional[list] = None, query: Optional[str] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 99 | Execute Asana operations. |
| ⚙️ | `_list_workspaces` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 182 | List all accessible workspaces. |
| ⚙️ | `_get_workspace` | `(self, workspace_gid: Optional[str] = None, **kwargs) -> ToolResult` | 199 | Get workspace details. |
| ⚙️ | `_list_projects` | `(self, workspace_gid: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 209 | List projects in workspace. |
| ⚙️ | `_get_project` | `(self, project_gid: Optional[str] = None, **kwargs) -> ToolResult` | 239 | Get project details. |
| ⚙️ | `_create_project` | `(self, workspace_gid: Optional[str] = None, name: Optional[str] = None, notes: Optional[str] = None, **kwargs) -> ToolResult` | 248 | Create a new project. |
| ⚙️ | `_update_project` | `(self, project_gid: Optional[str] = None, name: Optional[str] = None, notes: Optional[str] = None, **kwargs) -> ToolResult` | 275 | Update a project. |
| ⚙️ | `_delete_project` | `(self, project_gid: Optional[str] = None, **kwargs) -> ToolResult` | 299 | Delete a project. |
| ⚙️ | `_list_tasks` | `(self, project_gid: Optional[str] = None, section_gid: Optional[str] = None, assignee: Optional[str] = None, completed: Optional[bool] = None, limit: int = 100, **kwargs) -> ToolResult` | 312 | List tasks in a project or section. |
| ⚙️ | `_get_task` | `(self, task_gid: Optional[str] = None, **kwargs) -> ToolResult` | 350 | Get task details. |
| ⚙️ | `_create_task` | `(self, project_gid: Optional[str] = None, name: Optional[str] = None, notes: Optional[str] = None, due_on: Optional[str] = None, assignee: Optional[str] = None, **kwargs) -> ToolResult` | 359 | Create a new task. |
| ⚙️ | `_update_task` | `(self, task_gid: Optional[str] = None, name: Optional[str] = None, notes: Optional[str] = None, due_on: Optional[str] = None, assignee: Optional[str] = None, completed: Optional[bool] = None, **kwargs) -> ToolResult` | 389 | Update a task. |
| ⚙️ | `_delete_task` | `(self, task_gid: Optional[str] = None, **kwargs) -> ToolResult` | 422 | Delete a task. |
| ⚙️ | `_complete_task` | `(self, task_gid: Optional[str] = None, **kwargs) -> ToolResult` | 434 | Mark a task as complete. |
| ⚙️ | `_list_sections` | `(self, project_gid: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 451 | List sections in a project. |
| ⚙️ | `_create_section` | `(self, project_gid: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 481 | Create a section in a project. |
| ⚙️ | `_add_task_to_section` | `(self, section_gid: Optional[str] = None, task_gid: Optional[str] = None, **kwargs) -> ToolResult` | 504 | Add a task to a section. |
| ⚙️ | `_list_tags` | `(self, workspace_gid: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 529 | List tags in workspace. |
| ⚙️ | `_create_tag` | `(self, workspace_gid: Optional[str] = None, name: Optional[str] = None, **kwargs) -> ToolResult` | 559 | Create a tag. |
| ⚙️ | `_add_tag_to_task` | `(self, task_gid: Optional[str] = None, **kwargs) -> ToolResult` | 582 | Add a tag to a task. |
| ⚙️ | `_search` | `(self, workspace_gid: Optional[str] = None, query: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 608 | Search tasks in workspace. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 641 |  |

---

## `argus/tools/integrations/productivity_tools/atlassian.py`

### 🏛️ `class JiraTool(BaseTool)`  <sub>line 19</sub>

> Jira - Issue and project tracking.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, domain: Optional[str] = None, email: Optional[str] = None, api_token: Optional[str] = None, config: Optional[ToolConfig] = None)` | 42 |  |
| ⚙️ | `_get_session` | `(self)` | 62 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 78 | Make API request to Jira. |
| ⚙️ | `execute` | `(self, action: str = 'list_projects', project_key: Optional[str] = None, issue_key: Optional[str] = None, summary: Optional[str] = None, description: Optional[str] = None, issue_type: str = 'Task', priority: Optional[str] = None, assignee: Optional[str] = None, labels: Optional[list] = None, jql: Optional[str] = None, transition_id: Optional[str] = None, comment: Optional[str] = None, sprint_id: Optional[int] = None, limit: int = 50, **kwargs: Any) -> ToolResult` | 110 | Execute Jira operations. |
| ⚙️ | `_list_projects` | `(self, limit: int = 50, **kwargs) -> ToolResult` | 188 | List all projects. |
| ⚙️ | `_get_project` | `(self, project_key: Optional[str] = None, **kwargs) -> ToolResult` | 211 | Get project details. |
| ⚙️ | `_list_issues` | `(self, project_key: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 225 | List issues in a project. |
| ⚙️ | `_get_issue` | `(self, issue_key: Optional[str] = None, **kwargs) -> ToolResult` | 238 | Get issue details. |
| ⚙️ | `_create_issue` | `(self, project_key: Optional[str] = None, summary: Optional[str] = None, description: Optional[str] = None, issue_type: str = 'Task', priority: Optional[str] = None, assignee: Optional[str] = None, labels: Optional[list] = None, **kwargs) -> ToolResult` | 266 | Create a new issue. |
| ⚙️ | `_update_issue` | `(self, issue_key: Optional[str] = None, summary: Optional[str] = None, description: Optional[str] = None, priority: Optional[str] = None, labels: Optional[list] = None, **kwargs) -> ToolResult` | 318 | Update an issue. |
| ⚙️ | `_delete_issue` | `(self, issue_key: Optional[str] = None, **kwargs) -> ToolResult` | 364 | Delete an issue. |
| ⚙️ | `_assign_issue` | `(self, issue_key: Optional[str] = None, assignee: Optional[str] = None, **kwargs) -> ToolResult` | 380 | Assign an issue to a user. |
| ⚙️ | `_get_transitions` | `(self, issue_key: Optional[str] = None, **kwargs) -> ToolResult` | 401 | Get available transitions for an issue. |
| ⚙️ | `_transition_issue` | `(self, issue_key: Optional[str] = None, transition_id: Optional[str] = None, **kwargs) -> ToolResult` | 426 | Transition an issue to a new status. |
| ⚙️ | `_get_comments` | `(self, issue_key: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 451 | Get comments on an issue. |
| ⚙️ | `_add_comment` | `(self, issue_key: Optional[str] = None, comment: Optional[str] = None, **kwargs) -> ToolResult` | 491 | Add a comment to an issue. |
| ⚙️ | `_search` | `(self, jql: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 525 | Search issues using JQL. |
| ⚙️ | `_list_sprints` | `(self, **kwargs) -> ToolResult` | 565 | List sprints (requires board_id). |
| ⚙️ | `_add_to_sprint` | `(self, sprint_id: Optional[int] = None, issue_key: Optional[str] = None, **kwargs) -> ToolResult` | 601 | Add issue to a sprint. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 627 |  |

### 🏛️ `class ConfluenceTool(BaseTool)`  <sub>line 663</sub>

> Confluence - Documentation and knowledge management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, domain: Optional[str] = None, email: Optional[str] = None, api_token: Optional[str] = None, config: Optional[ToolConfig] = None)` | 685 |  |
| ⚙️ | `_get_session` | `(self)` | 702 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 718 | Make API request to Confluence. |
| ⚙️ | `execute` | `(self, action: str = 'list_spaces', space_key: Optional[str] = None, space_id: Optional[str] = None, page_id: Optional[str] = None, parent_id: Optional[str] = None, title: Optional[str] = None, content: Optional[str] = None, query: Optional[str] = None, limit: int = 25, **kwargs: Any) -> ToolResult` | 747 | Execute Confluence operations. |
| ⚙️ | `_list_spaces` | `(self, limit: int = 25, **kwargs) -> ToolResult` | 802 | List all spaces. |
| ⚙️ | `_get_space` | `(self, space_id: Optional[str] = None, **kwargs) -> ToolResult` | 825 | Get space details. |
| ⚙️ | `_list_pages` | `(self, space_id: Optional[str] = None, limit: int = 25, **kwargs) -> ToolResult` | 839 | List pages in a space. |
| ⚙️ | `_get_page` | `(self, page_id: Optional[str] = None, **kwargs) -> ToolResult` | 867 | Get page details with content. |
| ⚙️ | `_create_page` | `(self, space_id: Optional[str] = None, title: Optional[str] = None, content: Optional[str] = None, parent_id: Optional[str] = None, **kwargs) -> ToolResult` | 893 | Create a new page. |
| ⚙️ | `_update_page` | `(self, page_id: Optional[str] = None, title: Optional[str] = None, content: Optional[str] = None, **kwargs) -> ToolResult` | 928 | Update a page. |
| ⚙️ | `_delete_page` | `(self, page_id: Optional[str] = None, **kwargs) -> ToolResult` | 968 | Delete a page. |
| ⚙️ | `_search` | `(self, query: Optional[str] = None, limit: int = 25, **kwargs) -> ToolResult` | 985 | Search content. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 1025 |  |

---

## `argus/tools/integrations/productivity_tools/linear.py`

### 🏛️ `class LinearTool(BaseTool)`  <sub>line 18</sub>

> Linear - Issue tracking for engineering teams.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 43 |  |
| ⚙️ | `_get_session` | `(self)` | 59 | Get HTTP session with authentication. |
| ⚙️ | `_graphql` | `(self, query: str, variables: Optional[dict] = None) -> dict` | 70 | Execute GraphQL query. |
| ⚙️ | `execute` | `(self, action: str = 'list_teams', team_id: Optional[str] = None, project_id: Optional[str] = None, issue_id: Optional[str] = None, cycle_id: Optional[str] = None, title: Optional[str] = None, description: Optional[str] = None, priority: Optional[int] = None, state_id: Optional[str] = None, assignee_id: Optional[str] = None, label_ids: Optional[list] = None, query: Optional[str] = None, limit: int = 50, **kwargs: Any) -> ToolResult` | 91 | Execute Linear operations. |
| ⚙️ | `_list_teams` | `(self, limit: int = 50, **kwargs) -> ToolResult` | 174 | List all teams. |
| ⚙️ | `_get_team` | `(self, team_id: Optional[str] = None, **kwargs) -> ToolResult` | 206 | Get team details. |
| ⚙️ | `_list_issues` | `(self, team_id: Optional[str] = None, project_id: Optional[str] = None, cycle_id: Optional[str] = None, assignee_id: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 239 | List issues. |
| ⚙️ | `_get_issue` | `(self, issue_id: Optional[str] = None, **kwargs) -> ToolResult` | 308 | Get issue details. |
| ⚙️ | `_create_issue` | `(self, team_id: Optional[str] = None, title: Optional[str] = None, description: Optional[str] = None, priority: Optional[int] = None, state_id: Optional[str] = None, assignee_id: Optional[str] = None, project_id: Optional[str] = None, cycle_id: Optional[str] = None, label_ids: Optional[list] = None, **kwargs) -> ToolResult` | 374 | Create a new issue. |
| ⚙️ | `_update_issue` | `(self, issue_id: Optional[str] = None, title: Optional[str] = None, description: Optional[str] = None, priority: Optional[int] = None, state_id: Optional[str] = None, assignee_id: Optional[str] = None, label_ids: Optional[list] = None, **kwargs) -> ToolResult` | 437 | Update an issue. |
| ⚙️ | `_delete_issue` | `(self, issue_id: Optional[str] = None, **kwargs) -> ToolResult` | 493 | Delete (archive) an issue. |
| ⚙️ | `_list_projects` | `(self, team_id: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 519 | List projects. |
| ⚙️ | `_get_project` | `(self, project_id: Optional[str] = None, **kwargs) -> ToolResult` | 563 | Get project details. |
| ⚙️ | `_create_project` | `(self, team_id: Optional[str] = None, title: Optional[str] = None, description: Optional[str] = None, **kwargs) -> ToolResult` | 603 | Create a new project. |
| ⚙️ | `_list_cycles` | `(self, team_id: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 644 | List cycles. |
| ⚙️ | `_get_active_cycle` | `(self, team_id: Optional[str] = None, **kwargs) -> ToolResult` | 690 | Get the active cycle for a team. |
| ⚙️ | `_list_states` | `(self, team_id: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 722 | List workflow states. |
| ⚙️ | `_list_labels` | `(self, team_id: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 765 | List labels. |
| ⚙️ | `_create_label` | `(self, team_id: Optional[str] = None, title: Optional[str] = None, **kwargs) -> ToolResult` | 806 | Create a label. |
| ⚙️ | `_search` | `(self, query: Optional[str] = None, limit: int = 50, **kwargs) -> ToolResult` | 851 | Search issues. |
| ⚙️ | `_list_users` | `(self, limit: int = 50, **kwargs) -> ToolResult` | 899 | List users. |
| ⚙️ | `_me` | `(self, **kwargs) -> ToolResult` | 931 | Get current user. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 953 |  |

---

## `argus/tools/integrations/productivity_tools/notion.py`

### 🏛️ `class NotionTool(BaseTool)`  <sub>line 18</sub>

> Notion - Documentation and knowledge management.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 43 |  |
| ⚙️ | `_get_session` | `(self)` | 59 | Get HTTP session with authentication. |
| ⚙️ | `_request` | `(self, method: str, endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict` | 71 | Make API request to Notion. |
| ⚙️ | `execute` | `(self, action: str = 'search', page_id: Optional[str] = None, database_id: Optional[str] = None, parent_id: Optional[str] = None, parent_type: str = 'page_id', title: Optional[str] = None, content: Optional[str] = None, properties: Optional[dict] = None, filter: Optional[dict] = None, sorts: Optional[list] = None, query: Optional[str] = None, blocks: Optional[list] = None, limit: int = 100, **kwargs: Any) -> ToolResult` | 98 | Execute Notion operations. |
| ⚙️ | `_extract_title` | `(self, page: dict) -> str` | 170 | Extract title from page properties. |
| ⚙️ | `_get_page` | `(self, page_id: Optional[str] = None, **kwargs) -> ToolResult` | 193 | Get page details. |
| ⚙️ | `_create_page` | `(self, parent_id: Optional[str] = None, parent_type: str = 'page_id', title: Optional[str] = None, content: Optional[str] = None, properties: Optional[dict] = None, **kwargs) -> ToolResult` | 215 | Create a new page. |
| ⚙️ | `_update_page` | `(self, page_id: Optional[str] = None, properties: Optional[dict] = None, **kwargs) -> ToolResult` | 270 | Update page properties. |
| ⚙️ | `_archive_page` | `(self, page_id: Optional[str] = None, **kwargs) -> ToolResult` | 291 | Archive (delete) a page. |
| ⚙️ | `_get_database` | `(self, database_id: Optional[str] = None, **kwargs) -> ToolResult` | 308 | Get database details. |
| ⚙️ | `_query_database` | `(self, database_id: Optional[str] = None, filter: Optional[dict] = None, sorts: Optional[list] = None, limit: int = 100, **kwargs) -> ToolResult` | 336 | Query a database. |
| ⚙️ | `_create_database` | `(self, parent_id: Optional[str] = None, title: Optional[str] = None, properties: Optional[dict] = None, **kwargs) -> ToolResult` | 374 | Create a new database. |
| ⚙️ | `_get_blocks` | `(self, page_id: Optional[str] = None, limit: int = 100, **kwargs) -> ToolResult` | 404 | Get blocks (content) of a page. |
| ⚙️ | `_append_blocks` | `(self, page_id: Optional[str] = None, blocks: Optional[list] = None, content: Optional[str] = None, **kwargs) -> ToolResult` | 446 | Append blocks to a page. |
| ⚙️ | `_delete_block` | `(self, **kwargs) -> ToolResult` | 486 | Delete a block. |
| ⚙️ | `_search` | `(self, query: Optional[str] = None, filter: Optional[dict] = None, limit: int = 100, **kwargs) -> ToolResult` | 503 | Search pages and databases. |
| ⚙️ | `_list_users` | `(self, limit: int = 100, **kwargs) -> ToolResult` | 537 | List users. |
| ⚙️ | `_get_user` | `(self, **kwargs) -> ToolResult` | 560 | Get user details. |
| ⚙️ | `_me` | `(self, **kwargs) -> ToolResult` | 577 | Get the bot user. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 589 |  |

---

## `argus/tools/integrations/search/arxiv_tool.py`

### 🏛️ `class ArxivTool(BaseTool)`  <sub>line 17</sub>

> ArXiv paper search tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_results: int = 10, sort_by: str = 'relevance', config: Optional[ToolConfig] = None)` | 30 |  |
| ⚙️ | `execute` | `(self, query: str = '', max_results: Optional[int] = None, **kwargs: Any) -> ToolResult` | 40 | Search ArXiv for papers. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 98 |  |

---

## `argus/tools/integrations/search/brave.py`

### 🏛️ `class BraveTool(BaseTool)`  <sub>line 18</sub>

> Brave Search tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 33 |  |
| ⚙️ | `execute` | `(self, query: str = '', count: int = 10, freshness: Optional[str] = None, **kwargs: Any) -> ToolResult` | 41 | Search with Brave. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 101 |  |

---

## `argus/tools/integrations/search/duckduckgo.py`

### 🏛️ `class DuckDuckGoTool(BaseTool)`  <sub>line 17</sub>

> DuckDuckGo search tool - free, no API key required.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, max_results: int = 10, region: str = 'wt-wt', safesearch: str = 'moderate', config: Optional[ToolConfig] = None)` | 31 |  |
| ⚙️ | `execute` | `(self, query: str = '', max_results: Optional[int] = None, **kwargs: Any) -> ToolResult` | 43 | Search DuckDuckGo. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 99 |  |

---

## `argus/tools/integrations/search/exa.py`

### 🏛️ `class ExaTool(BaseTool)`  <sub>line 18</sub>

> Exa neural search tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 31 |  |
| ⚙️ | `execute` | `(self, query: str = '', num_results: int = 10, use_autoprompt: bool = True, include_domains: Optional[list[str]] = None, exclude_domains: Optional[list[str]] = None, **kwargs: Any) -> ToolResult` | 39 | Search with Exa. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 110 |  |

---

## `argus/tools/integrations/search/tavily.py`

### 🏛️ `class TavilyTool(BaseTool)`  <sub>line 18</sub>

> Tavily AI-optimized search tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, search_depth: str = 'basic', config: Optional[ToolConfig] = None)` | 31 |  |
| ⚙️ | `execute` | `(self, query: str = '', search_depth: Optional[str] = None, include_answer: bool = True, include_images: bool = False, max_results: int = 5, **kwargs: Any) -> ToolResult` | 41 | Search with Tavily. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 108 |  |

---

## `argus/tools/integrations/search/wikipedia.py`

### 🏛️ `class WikipediaTool(BaseTool)`  <sub>line 17</sub>

> Wikipedia search and article retrieval tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, language: str = 'en', sentences: int = 5, config: Optional[ToolConfig] = None)` | 30 |  |
| ⚙️ | `execute` | `(self, query: str = '', action: str = 'search', sentences: Optional[int] = None, **kwargs: Any) -> ToolResult` | 40 | Search or retrieve Wikipedia content. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 113 |  |

---

## `argus/tools/integrations/vectordb/chroma.py`

### 🏛️ `class ChromaTool(BaseTool)`  <sub>line 20</sub>

> Chroma - Open-source embedding database.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, persist_directory: Optional[str] = None, host: Optional[str] = None, port: int = 8000, embedding_function: Optional[str] = 'default', config: Optional[ToolConfig] = None)` | 42 |  |
| ⚙️ | `_get_client` | `(self)` | 62 | Lazy-load Chroma client. |
| ⚙️ | `_get_embedding_function` | `(self)` | 83 | Get embedding function. |
| ⚙️ | `execute` | `(self, action: str = 'list_collections', collection: Optional[str] = None, documents: Optional[list] = None, embeddings: Optional[list] = None, metadatas: Optional[list] = None, ids: Optional[list] = None, query_text: Optional[str] = None, query_embedding: Optional[list] = None, n_results: int = 10, where: Optional[dict] = None, where_document: Optional[dict] = None, include: Optional[list] = None, **kwargs: Any) -> ToolResult` | 105 | Execute Chroma operations. |
| ⚙️ | `_get_collection_obj` | `(self, collection: str)` | 162 | Get or create a collection. |
| ⚙️ | `_add` | `(self, collection: Optional[str] = None, documents: Optional[list] = None, embeddings: Optional[list] = None, metadatas: Optional[list] = None, ids: Optional[list] = None, **kwargs) -> ToolResult` | 172 | Add documents to a collection. |
| ⚙️ | `_query` | `(self, collection: Optional[str] = None, query_text: Optional[str] = None, query_embedding: Optional[list] = None, n_results: int = 10, where: Optional[dict] = None, where_document: Optional[dict] = None, include: Optional[list] = None, **kwargs) -> ToolResult` | 209 | Query the collection for similar documents. |
| ⚙️ | `_get` | `(self, collection: Optional[str] = None, ids: Optional[list] = None, where: Optional[dict] = None, include: Optional[list] = None, limit: int = 100, **kwargs) -> ToolResult` | 264 | Get documents by ID or filter. |
| ⚙️ | `_update` | `(self, collection: Optional[str] = None, ids: Optional[list] = None, documents: Optional[list] = None, embeddings: Optional[list] = None, metadatas: Optional[list] = None, **kwargs) -> ToolResult` | 307 | Update documents in a collection. |
| ⚙️ | `_delete` | `(self, collection: Optional[str] = None, ids: Optional[list] = None, where: Optional[dict] = None, where_document: Optional[dict] = None, **kwargs) -> ToolResult` | 339 | Delete documents from a collection. |
| ⚙️ | `_create_collection` | `(self, collection: Optional[str] = None, metadata: Optional[dict] = None, **kwargs) -> ToolResult` | 368 | Create a new collection. |
| ⚙️ | `_delete_collection` | `(self, collection: Optional[str] = None, **kwargs) -> ToolResult` | 392 | Delete a collection. |
| ⚙️ | `_list_collections` | `(self, **kwargs) -> ToolResult` | 409 | List all collections. |
| ⚙️ | `_get_collection` | `(self, collection: Optional[str] = None, **kwargs) -> ToolResult` | 422 | Get collection info. |
| ⚙️ | `_count` | `(self, collection: Optional[str] = None, **kwargs) -> ToolResult` | 439 | Get document count in a collection. |
| ⚙️ | `_peek` | `(self, collection: Optional[str] = None, limit: int = 10, **kwargs) -> ToolResult` | 455 | Peek at documents in a collection. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 483 |  |

---

## `argus/tools/integrations/vectordb/mongodb.py`

### 🏛️ `class MongoDBTool(BaseTool)`  <sub>line 19</sub>

> MongoDB - Flexible document database with vector search.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, connection_string: Optional[str] = None, database: Optional[str] = None, config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_client` | `(self)` | 59 | Lazy-load MongoDB client. |
| ⚙️ | `execute` | `(self, action: str = 'list_databases', database: Optional[str] = None, collection: Optional[str] = None, documents: Optional[list] = None, document: Optional[dict] = None, query: Optional[dict] = None, update: Optional[dict] = None, projection: Optional[dict] = None, sort: Optional[list] = None, limit: int = 100, skip: int = 0, vector: Optional[list] = None, index_name: str = 'vector_index', num_candidates: int = 100, pipeline: Optional[list] = None, **kwargs: Any) -> ToolResult` | 69 | Execute MongoDB operations. |
| ⚙️ | `_insert` | `(self, database: str, collection: Optional[str] = None, documents: Optional[list] = None, **kwargs) -> ToolResult` | 141 | Insert multiple documents. |
| ⚙️ | `_insert_one` | `(self, database: str, collection: Optional[str] = None, document: Optional[dict] = None, **kwargs) -> ToolResult` | 172 | Insert a single document. |
| ⚙️ | `_find` | `(self, database: str, collection: Optional[str] = None, query: Optional[dict] = None, projection: Optional[dict] = None, sort: Optional[list] = None, limit: int = 100, skip: int = 0, **kwargs) -> ToolResult` | 200 | Find documents matching query. |
| ⚙️ | `_find_one` | `(self, database: str, collection: Optional[str] = None, query: Optional[dict] = None, projection: Optional[dict] = None, **kwargs) -> ToolResult` | 241 | Find a single document. |
| ⚙️ | `_update` | `(self, database: str, collection: Optional[str] = None, query: Optional[dict] = None, update: Optional[dict] = None, **kwargs) -> ToolResult` | 269 | Update multiple documents. |
| ⚙️ | `_update_one` | `(self, database: str, collection: Optional[str] = None, query: Optional[dict] = None, update: Optional[dict] = None, **kwargs) -> ToolResult` | 306 | Update a single document. |
| ⚙️ | `_delete` | `(self, database: str, collection: Optional[str] = None, query: Optional[dict] = None, **kwargs) -> ToolResult` | 341 | Delete multiple documents. |
| ⚙️ | `_delete_one` | `(self, database: str, collection: Optional[str] = None, query: Optional[dict] = None, **kwargs) -> ToolResult` | 364 | Delete a single document. |
| ⚙️ | `_count` | `(self, database: str, collection: Optional[str] = None, query: Optional[dict] = None, **kwargs) -> ToolResult` | 387 | Count documents matching query. |
| ⚙️ | `_aggregate` | `(self, database: str, collection: Optional[str] = None, pipeline: Optional[list] = None, **kwargs) -> ToolResult` | 410 | Run aggregation pipeline. |
| ⚙️ | `_vector_search` | `(self, database: str, collection: Optional[str] = None, vector: Optional[list] = None, index_name: str = 'vector_index', limit: int = 10, num_candidates: int = 100, projection: Optional[dict] = None, **kwargs) -> ToolResult` | 441 | Run Atlas Vector Search. |
| ⚙️ | `_create_index` | `(self, database: str, collection: Optional[str] = None, **kwargs) -> ToolResult` | 497 | Create an index on a collection. |
| ⚙️ | `_create_vector_index` | `(self, database: str, collection: Optional[str] = None, index_name: str = 'vector_index', **kwargs) -> ToolResult` | 523 | Create a vector search index (Atlas). |
| ⚙️ | `_list_indexes` | `(self, database: str, collection: Optional[str] = None, **kwargs) -> ToolResult` | 563 | List indexes on a collection. |
| ⚙️ | `_drop_index` | `(self, database: str, collection: Optional[str] = None, **kwargs) -> ToolResult` | 586 | Drop an index. |
| ⚙️ | `_list_collections` | `(self, database: str, **kwargs) -> ToolResult` | 612 | List collections in database. |
| ⚙️ | `_create_collection` | `(self, database: str, collection: Optional[str] = None, **kwargs) -> ToolResult` | 629 | Create a collection. |
| ⚙️ | `_drop_collection` | `(self, database: str, collection: Optional[str] = None, **kwargs) -> ToolResult` | 650 | Drop a collection. |
| ⚙️ | `_list_databases` | `(self, **kwargs) -> ToolResult` | 671 | List all databases. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 682 |  |

---

## `argus/tools/integrations/vectordb/pinecone_tool.py`

### 🏛️ `class PineconeTool(BaseTool)`  <sub>line 19</sub>

> Pinecone - Vector database for semantic search.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, environment: Optional[str] = None, index_name: Optional[str] = None, config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_client` | `(self)` | 59 | Lazy-load Pinecone client. |
| ⚙️ | `_get_index` | `(self, index_name: Optional[str] = None)` | 69 | Get index instance. |
| ⚙️ | `execute` | `(self, action: str = 'list_indexes', index_name: Optional[str] = None, vectors: Optional[list] = None, vector: Optional[list] = None, ids: Optional[list] = None, top_k: int = 10, namespace: str = '', filter: Optional[dict] = None, include_values: bool = False, include_metadata: bool = True, dimension: int = 1536, metric: str = 'cosine', **kwargs: Any) -> ToolResult` | 78 | Execute Pinecone operations. |
| ⚙️ | `_upsert` | `(self, index_name: Optional[str] = None, vectors: Optional[list] = None, namespace: str = '', **kwargs) -> ToolResult` | 134 | Upsert vectors into the index. |
| ⚙️ | `_query` | `(self, index_name: Optional[str] = None, vector: Optional[list] = None, top_k: int = 10, namespace: str = '', filter: Optional[dict] = None, include_values: bool = False, include_metadata: bool = True, **kwargs) -> ToolResult` | 172 | Query the index for similar vectors. |
| ⚙️ | `_fetch` | `(self, index_name: Optional[str] = None, ids: Optional[list] = None, namespace: str = '', **kwargs) -> ToolResult` | 219 | Fetch vectors by ID. |
| ⚙️ | `_delete` | `(self, index_name: Optional[str] = None, ids: Optional[list] = None, namespace: str = '', delete_all: bool = False, filter: Optional[dict] = None, **kwargs) -> ToolResult` | 246 | Delete vectors from the index. |
| ⚙️ | `_update` | `(self, index_name: Optional[str] = None, id: Optional[str] = None, values: Optional[list] = None, set_metadata: Optional[dict] = None, namespace: str = '', **kwargs) -> ToolResult` | 276 | Update a vector. |
| ⚙️ | `_list_indexes` | `(self, **kwargs) -> ToolResult` | 304 | List all indexes. |
| ⚙️ | `_create_index` | `(self, index_name: Optional[str] = None, dimension: int = 1536, metric: str = 'cosine', **kwargs) -> ToolResult` | 323 | Create a new index. |
| ⚙️ | `_delete_index` | `(self, index_name: Optional[str] = None, **kwargs) -> ToolResult` | 355 | Delete an index. |
| ⚙️ | `_describe_index` | `(self, index_name: Optional[str] = None, **kwargs) -> ToolResult` | 372 | Describe an index. |
| ⚙️ | `_describe_index_stats` | `(self, index_name: Optional[str] = None, **kwargs) -> ToolResult` | 393 | Get index statistics. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 409 |  |

---

## `argus/tools/integrations/vectordb/qdrant.py`

### 🏛️ `class QdrantTool(BaseTool)`  <sub>line 19</sub>

> Qdrant - Vector search engine.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, host: Optional[str] = None, port: int = 6333, api_key: Optional[str] = None, url: Optional[str] = None, config: Optional[ToolConfig] = None)` | 41 |  |
| ⚙️ | `_get_client` | `(self)` | 60 | Lazy-load Qdrant client. |
| ⚙️ | `execute` | `(self, action: str = 'list_collections', collection: Optional[str] = None, vectors: Optional[list] = None, query_vector: Optional[list] = None, ids: Optional[list] = None, limit: int = 10, filter: Optional[dict] = None, with_payload: bool = True, with_vectors: bool = False, dimension: int = 1536, distance: str = 'Cosine', **kwargs: Any) -> ToolResult` | 74 | Execute Qdrant operations. |
| ⚙️ | `_upsert` | `(self, collection: Optional[str] = None, vectors: Optional[list] = None, **kwargs) -> ToolResult` | 128 | Upsert points into a collection. |
| ⚙️ | `_search` | `(self, collection: Optional[str] = None, query_vector: Optional[list] = None, limit: int = 10, filter: Optional[dict] = None, with_payload: bool = True, with_vectors: bool = False, **kwargs) -> ToolResult` | 170 | Search for similar vectors. |
| ⚙️ | `_retrieve` | `(self, collection: Optional[str] = None, ids: Optional[list] = None, with_payload: bool = True, with_vectors: bool = False, **kwargs) -> ToolResult` | 224 | Retrieve points by ID. |
| ⚙️ | `_delete` | `(self, collection: Optional[str] = None, ids: Optional[list] = None, filter: Optional[dict] = None, **kwargs) -> ToolResult` | 270 | Delete points from a collection. |
| ⚙️ | `_scroll` | `(self, collection: Optional[str] = None, limit: int = 100, with_payload: bool = True, with_vectors: bool = False, offset: Optional[str] = None, **kwargs) -> ToolResult` | 314 | Scroll through all points. |
| ⚙️ | `_create_collection` | `(self, collection: Optional[str] = None, dimension: int = 1536, distance: str = 'Cosine', **kwargs) -> ToolResult` | 353 | Create a new collection. |
| ⚙️ | `_delete_collection` | `(self, collection: Optional[str] = None, **kwargs) -> ToolResult` | 389 | Delete a collection. |
| ⚙️ | `_list_collections` | `(self, **kwargs) -> ToolResult` | 406 | List all collections. |
| ⚙️ | `_get_collection` | `(self, collection: Optional[str] = None, **kwargs) -> ToolResult` | 419 | Get collection info. |
| ⚙️ | `_count` | `(self, collection: Optional[str] = None, **kwargs) -> ToolResult` | 438 | Count points in a collection. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 455 |  |

---

## `argus/tools/integrations/web/jina_reader.py`

### 🏛️ `class JinaReaderTool(BaseTool)`  <sub>line 17</sub>

> Jina Reader - convert web pages to clean markdown.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, api_key: Optional[str] = None, config: Optional[ToolConfig] = None)` | 32 |  |
| ⚙️ | `execute` | `(self, url: str = '', **kwargs: Any) -> ToolResult` | 41 | Convert URL to markdown. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 87 |  |

---

## `argus/tools/integrations/web/requests_tool.py`

### 🏛️ `class RequestsTool(BaseTool)`  <sub>line 17</sub>

> HTTP requests tool for fetching web content.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, timeout: float = 30.0, headers: Optional[dict] = None, config: Optional[ToolConfig] = None)` | 30 |  |
| ⚙️ | `execute` | `(self, url: str = '', method: str = 'GET', headers: Optional[dict] = None, data: Optional[dict] = None, json: Optional[dict] = None, params: Optional[dict] = None, **kwargs: Any) -> ToolResult` | 40 | Make HTTP request. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 109 |  |

---

## `argus/tools/integrations/web/scraper.py`

### 🏛️ `class WebScraperTool(BaseTool)`  <sub>line 17</sub>

> Web scraping tool using BeautifulSoup.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, timeout: float = 30.0, config: Optional[ToolConfig] = None)` | 30 |  |
| ⚙️ | `execute` | `(self, url: str = '', extract: str = 'text', selector: Optional[str] = None, **kwargs: Any) -> ToolResult` | 38 | Scrape web page. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 115 |  |

---

## `argus/tools/integrations/web/youtube.py`

### 🏛️ `class YouTubeTool(BaseTool)`  <sub>line 18</sub>

> YouTube transcript extraction tool.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self, config: Optional[ToolConfig] = None)` | 31 |  |
| ⚙️ | `_extract_video_id` | `(self, url_or_id: str) -> Optional[str]` | 34 | Extract video ID from URL or return as-is. |
| ⚙️ | `execute` | `(self, video_url: str = '', language: str = 'en', **kwargs: Any) -> ToolResult` | 46 | Get YouTube transcript. |
| ⚙️ | `get_schema` | `(self) -> dict[str, Any]` | 111 |  |

---

## `argus/tools/registry.py`

### 🏛️ `class ToolRegistry()`  <sub>line 25</sub>

> Central registry for ARGUS tools.  

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| ⚙️ | `__init__` | `(self)` | 45 | Initialize an empty registry. |
| ⚙️ | `register` | `(self, tool: BaseTool) -> None` | 52 | Register a tool instance. |
| ⚙️ | `register_class` | `(self, tool_class: Type[BaseTool], **init_kwargs: Any) -> None` | 68 | Register a tool class for lazy instantiation. |
| ⚙️ | `unregister` | `(self, name: str) -> bool` | 86 | Unregister a tool by name. |
| ⚙️ | `get` | `(self, name: str) -> Optional[BaseTool]` | 102 | Get a tool by name. |
| ⚙️ | `get_or_raise` | `(self, name: str) -> BaseTool` | 114 | Get a tool by name, raising if not found. |
| ⚙️ | `has` | `(self, name: str) -> bool` | 131 | Check if a tool is registered. |
| ⚙️ | `list_all` | `(self) -> list[str]` | 143 | Get list of all registered tool names. |
| ⚙️ | `list_by_category` | `(self, category: ToolCategory) -> list[BaseTool]` | 152 | Get all tools in a category. |
| ⚙️ | `get_schemas` | `(self) -> list[dict[str, Any]]` | 167 | Get JSON schemas for all registered tools. |
| ⚙️ | `get_stats` | `(self) -> dict[str, Any]` | 176 | Get execution statistics for all tools. |
| ⚙️ | `clear` | `(self) -> None` | 191 | Unregister all tools. |
| ⚙️ | `__len__` | `(self) -> int` | 197 | Get number of registered tools. |
| ⚙️ | `__contains__` | `(self, name: str) -> bool` | 202 | Check if tool is registered. |
| ⚙️ | `__iter__` | `(self)` | 206 | Iterate over tools. |

### 🔧 Module-level Functions

| Kind | Name | Signature | Line | Docstring |
|------|------|-----------|-----:|-----------|
| 🔧 | `get_default_registry` | `() -> ToolRegistry` | 220 | Get the global default tool registry. |
| 🔧 | `register_tool` | `(tool: BaseTool) -> None` | 255 | Register a tool with the default registry. |
| 🔧 | `get_tool` | `(name: str) -> Optional[BaseTool]` | 266 | Get a tool from the default registry. |
| 🔧 | `list_tools` | `() -> list[str]` | 278 | List all tools in the default registry. |

---

## Flat Symbol Index

| File | Kind | Qualname | Line |
|------|------|----------|-----:|
| `argus/agents/base.py` | 🏛️ class | `AgentRole` | 26 |
| `argus/agents/base.py` | 🏛️ class | `AgentConfig` | 35 |
| `argus/agents/base.py` | 🏛️ class | `AgentResponse` | 76 |
| `argus/agents/base.py` | ⚙️ method | `AgentResponse.failed` | 96 |
| `argus/agents/base.py` | 🏛️ class | `BaseAgent` | 101 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.__init__` | 113 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.name` | 141 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.role` | 146 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.get_system_prompt` | 151 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.act` | 161 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.generate` | 178 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.log_action` | 209 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.get_history` | 230 |
| `argus/agents/base.py` | ⚙️ method | `BaseAgent.__repr__` | 234 |
| `argus/agents/jury.py` | 🏛️ class | `JuryConfig` | 32 |
| `argus/agents/jury.py` | 🏛️ class | `Verdict` | 52 |
| `argus/agents/jury.py` | ⚙️ method | `Verdict.to_dict` | 77 |
| `argus/agents/jury.py` | 🏛️ class | `Jury` | 92 |
| `argus/agents/jury.py` | ⚙️ method | `Jury.__init__` | 108 |
| `argus/agents/jury.py` | ⚙️ method | `Jury.get_system_prompt` | 122 |
| `argus/agents/jury.py` | ⚙️ method | `Jury.act` | 155 |
| `argus/agents/jury.py` | ⚙️ method | `Jury.evaluate` | 203 |
| `argus/agents/jury.py` | ⚙️ method | `Jury.evaluate_all` | 284 |
| `argus/agents/jury.py` | ⚙️ method | `Jury._generate_reasoning` | 306 |
| `argus/agents/jury.py` | ⚙️ method | `Jury.compute_disagreement` | 335 |
| `argus/agents/moderator.py` | 🏛️ class | `ModeratorConfig` | 35 |
| `argus/agents/moderator.py` | 🏛️ class | `DebateAgenda` | 62 |
| `argus/agents/moderator.py` | ⚙️ method | `DebateAgenda.to_dict` | 85 |
| `argus/agents/moderator.py` | 🏛️ class | `Moderator` | 99 |
| `argus/agents/moderator.py` | ⚙️ method | `Moderator.__init__` | 117 |
| `argus/agents/moderator.py` | ⚙️ method | `Moderator.get_system_prompt` | 132 |
| `argus/agents/moderator.py` | ⚙️ method | `Moderator.act` | 152 |
| `argus/agents/moderator.py` | ⚙️ method | `Moderator.create_agenda` | 201 |
| `argus/agents/moderator.py` | ⚙️ method | `Moderator.advance_round` | 281 |
| `argus/agents/moderator.py` | ⚙️ method | `Moderator.should_stop` | 319 |
| `argus/agents/moderator.py` | ⚙️ method | `Moderator._evaluate_round` | 357 |
| `argus/agents/refuter.py` | 🏛️ class | `RefuterConfig` | 33 |
| `argus/agents/refuter.py` | 🏛️ class | `Refuter` | 53 |
| `argus/agents/refuter.py` | ⚙️ method | `Refuter.__init__` | 70 |
| `argus/agents/refuter.py` | ⚙️ method | `Refuter.get_system_prompt` | 84 |
| `argus/agents/refuter.py` | ⚙️ method | `Refuter.act` | 122 |
| `argus/agents/refuter.py` | ⚙️ method | `Refuter.generate_rebuttals` | 170 |
| `argus/agents/refuter.py` | ⚙️ method | `Refuter.find_contradictions` | 273 |
| `argus/agents/refuter.py` | ⚙️ method | `Refuter._strip_json` | 335 |
| `argus/agents/specialist.py` | 🏛️ class | `SpecialistConfig` | 34 |
| `argus/agents/specialist.py` | 🏛️ class | `Specialist` | 59 |
| `argus/agents/specialist.py` | ⚙️ method | `Specialist.__init__` | 76 |
| `argus/agents/specialist.py` | ⚙️ method | `Specialist.domain` | 96 |
| `argus/agents/specialist.py` | ⚙️ method | `Specialist.get_system_prompt` | 100 |
| `argus/agents/specialist.py` | ⚙️ method | `Specialist.act` | 128 |
| `argus/agents/specialist.py` | ⚙️ method | `Specialist.gather_evidence` | 182 |
| `argus/agents/specialist.py` | ⚙️ method | `Specialist.evaluate_chunk` | 286 |
| `argus/agents/specialist.py` | ⚙️ method | `Specialist._create_evidence` | 350 |
| `argus/cdag/edges.py` | 🔧 function | `generate_edge_id` | 22 |
| `argus/cdag/edges.py` | 🏛️ class | `EdgeType` | 27 |
| `argus/cdag/edges.py` | 🏛️ class | `EdgePolarity` | 39 |
| `argus/cdag/edges.py` | 🏛️ class | `Edge` | 59 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.model_post_init` | 156 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.effective_weight` | 166 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.signed_weight` | 176 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.is_supporting` | 189 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.is_attacking` | 195 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.update_weight` | 199 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.__hash__` | 225 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.__eq__` | 229 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.to_tuple` | 235 |
| `argus/cdag/edges.py` | ⚙️ method | `Edge.to_prov_dict` | 239 |
| `argus/cdag/edges.py` | 🔧 function | `create_support_edge` | 258 |
| `argus/cdag/edges.py` | 🔧 function | `create_attack_edge` | 289 |
| `argus/cdag/edges.py` | 🔧 function | `create_rebuttal_edge` | 320 |
| `argus/cdag/graph.py` | 🏛️ class | `CDAG` | 38 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.__init__` | 71 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.add_node` | 107 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.add_proposition` | 134 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.add_evidence` | 146 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.add_rebuttal` | 198 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_node` | 239 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_proposition` | 251 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.remove_node` | 266 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.add_edge` | 299 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_edge` | 329 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.remove_edge` | 341 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_incoming_edges` | 373 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_outgoing_edges` | 400 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_supporting_evidence` | 423 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_attacking_evidence` | 447 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_all_propositions` | 471 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_all_evidence` | 484 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_support_path` | 499 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_support_path._trace_path` | 516 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_attack_path` | 537 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.get_attack_path._trace_path` | 554 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.compute_support_score` | 577 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.compute_attack_score` | 604 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.compute_net_influence` | 630 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.num_nodes` | 645 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.num_edges` | 650 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.num_propositions` | 655 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.num_evidence` | 660 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.__repr__` | 664 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.summary` | 673 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.to_networkx` | 690 |
| `argus/cdag/graph.py` | ⚙️ method | `CDAG.clear` | 699 |
| `argus/cdag/nodes.py` | 🔧 function | `generate_node_id` | 22 |
| `argus/cdag/nodes.py` | 🏛️ class | `NodeStatus` | 27 |
| `argus/cdag/nodes.py` | 🏛️ class | `NodeBase` | 38 |
| `argus/cdag/nodes.py` | ⚙️ method | `NodeBase.score` | 104 |
| `argus/cdag/nodes.py` | ⚙️ method | `NodeBase.score` | 109 |
| `argus/cdag/nodes.py` | ⚙️ method | `NodeBase.validate_confidence` | 115 |
| `argus/cdag/nodes.py` | ⚙️ method | `NodeBase.update_status` | 119 |
| `argus/cdag/nodes.py` | ⚙️ method | `NodeBase.__hash__` | 129 |
| `argus/cdag/nodes.py` | ⚙️ method | `NodeBase.__eq__` | 133 |
| `argus/cdag/nodes.py` | 🏛️ class | `Proposition` | 140 |
| `argus/cdag/nodes.py` | ⚙️ method | `Proposition.log_odds_prior` | 192 |
| `argus/cdag/nodes.py` | ⚙️ method | `Proposition.log_odds_posterior` | 200 |
| `argus/cdag/nodes.py` | ⚙️ method | `Proposition.update_posterior` | 206 |
| `argus/cdag/nodes.py` | 🏛️ class | `EvidenceType` | 230 |
| `argus/cdag/nodes.py` | 🏛️ class | `Evidence` | 239 |
| `argus/cdag/nodes.py` | ⚙️ method | `Evidence.effective_weight` | 305 |
| `argus/cdag/nodes.py` | ⚙️ method | `Evidence.is_supporting` | 315 |
| `argus/cdag/nodes.py` | ⚙️ method | `Evidence.is_attacking` | 321 |
| `argus/cdag/nodes.py` | ⚙️ method | `Evidence.add_citation` | 325 |
| `argus/cdag/nodes.py` | 🏛️ class | `Rebuttal` | 352 |
| `argus/cdag/nodes.py` | 🏛️ class | `Finding` | 387 |
| `argus/cdag/nodes.py` | 🏛️ class | `Assumption` | 421 |
| `argus/cdag/propagation.py` | 🏛️ class | `PropagationConfig` | 34 |
| `argus/cdag/propagation.py` | 🔧 function | `sigmoid` | 56 |
| `argus/cdag/propagation.py` | 🔧 function | `squash` | 71 |
| `argus/cdag/propagation.py` | 🔧 function | `log_odds` | 99 |
| `argus/cdag/propagation.py` | 🔧 function | `from_log_odds` | 113 |
| `argus/cdag/propagation.py` | 🔧 function | `compute_log_likelihood_ratio` | 126 |
| `argus/cdag/propagation.py` | 🔧 function | `propagate_influence` | 159 |
| `argus/cdag/propagation.py` | 🔧 function | `compute_posterior` | 277 |
| `argus/cdag/propagation.py` | 🔧 function | `compute_all_posteriors` | 359 |
| `argus/cdag/propagation.py` | 🔧 function | `compute_disagreement_index` | 382 |
| `argus/cdag/propagation.py` | 🔧 function | `get_counter_evidence` | 413 |
| `argus/cli.py` | 🔧 function | `setup_parser` | 29 |
| `argus/cli.py` | 🔧 function | `cmd_debate` | 432 |
| `argus/cli.py` | 🔧 function | `cmd_evaluate` | 476 |
| `argus/cli.py` | 🔧 function | `cmd_ingest` | 507 |
| `argus/cli.py` | 🔧 function | `cmd_providers` | 557 |
| `argus/cli.py` | 🔧 function | `cmd_benchmark` | 597 |
| `argus/cli.py` | 🔧 function | `cmd_datasets` | 656 |
| `argus/cli.py` | 🔧 function | `cmd_report` | 696 |
| `argus/cli.py` | 🔧 function | `cmd_tools` | 746 |
| `argus/cli.py` | 🔧 function | `cmd_embeddings` | 799 |
| `argus/cli.py` | 🔧 function | `cmd_score` | 883 |
| `argus/cli.py` | 🔧 function | `cmd_config` | 927 |
| `argus/cli.py` | 🔧 function | `cmd_openapi` | 949 |
| `argus/cli.py` | 🔧 function | `cmd_cache` | 1007 |
| `argus/cli.py` | 🔧 function | `cmd_compress` | 1065 |
| `argus/cli.py` | 🔧 function | `cmd_visualize` | 1135 |
| `argus/cli.py` | 🔧 function | `main` | 1215 |
| `argus/core/config.py` | 🏛️ class | `LLMProviderConfig` | 50 |
| `argus/core/config.py` | ⚙️ method | `LLMProviderConfig.has_provider` | 103 |
| `argus/core/config.py` | ⚙️ method | `LLMProviderConfig.get_available_providers` | 140 |
| `argus/core/config.py` | 🏛️ class | `EmbeddingProviderConfig` | 151 |
| `argus/core/config.py` | ⚙️ method | `EmbeddingProviderConfig.has_provider` | 199 |
| `argus/core/config.py` | ⚙️ method | `EmbeddingProviderConfig.get_available_providers` | 224 |
| `argus/core/config.py` | 🏛️ class | `RetrievalConfig` | 235 |
| `argus/core/config.py` | 🏛️ class | `ChunkingConfig` | 290 |
| `argus/core/config.py` | ⚙️ method | `ChunkingConfig.validate_overlap` | 339 |
| `argus/core/config.py` | ⚙️ method | `ChunkingConfig.validate_chunk_params` | 346 |
| `argus/core/config.py` | 🏛️ class | `CalibrationConfig` | 361 |
| `argus/core/config.py` | 🏛️ class | `ProvenanceConfig` | 399 |
| `argus/core/config.py` | 🏛️ class | `ArgusConfig` | 439 |
| `argus/core/config.py` | ⚙️ method | `ArgusConfig.validate_config` | 536 |
| `argus/core/config.py` | ⚙️ method | `ArgusConfig.configure_logging` | 546 |
| `argus/core/config.py` | ⚙️ method | `ArgusConfig.get_model_for_provider` | 554 |
| `argus/core/config.py` | 🔧 function | `get_config` | 586 |
| `argus/core/config.py` | 🔧 function | `reset_config` | 608 |
| `argus/core/config.py` | 🔧 function | `set_config` | 620 |
| `argus/core/context_caching.py` | 🏛️ class | `CacheEntry` | 30 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheEntry.is_expired` | 42 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheEntry.age_seconds` | 49 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheEntry.touch` | 53 |
| `argus/core/context_caching.py` | 🏛️ class | `CacheStats` | 60 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheStats.hit_rate` | 72 |
| `argus/core/context_caching.py` | 🏛️ class | `CacheBackend` | 78 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheBackend.get` | 82 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheBackend.set` | 87 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheBackend.delete` | 92 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheBackend.exists` | 97 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheBackend.clear` | 102 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheBackend.keys` | 107 |
| `argus/core/context_caching.py` | ⚙️ method | `CacheBackend.size` | 112 |
| `argus/core/context_caching.py` | 🏛️ class | `MemoryBackend` | 117 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.__init__` | 120 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.get` | 127 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.set` | 136 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.delete` | 157 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.exists` | 165 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.clear` | 168 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.keys` | 175 |
| `argus/core/context_caching.py` | ⚙️ method | `MemoryBackend.size` | 178 |
| `argus/core/context_caching.py` | 🏛️ class | `FileBackend` | 182 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.__init__` | 185 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend._load_index` | 193 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend._save_index` | 202 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend._get_entry_path` | 207 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.get` | 213 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.set` | 238 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.delete` | 257 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.exists` | 270 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.clear` | 273 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.keys` | 286 |
| `argus/core/context_caching.py` | ⚙️ method | `FileBackend.size` | 289 |
| `argus/core/context_caching.py` | 🏛️ class | `RedisBackend` | 293 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.__init__` | 296 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend._make_key` | 318 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.get` | 321 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.set` | 339 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.delete` | 350 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.exists` | 353 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.clear` | 356 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.keys` | 362 |
| `argus/core/context_caching.py` | ⚙️ method | `RedisBackend.size` | 366 |
| `argus/core/context_caching.py` | 🏛️ class | `ContextCache` | 370 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.__init__` | 387 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache._make_key` | 402 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache._estimate_size` | 408 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.get` | 415 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.set` | 446 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.delete` | 482 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.exists` | 487 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.clear` | 501 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.get_stats` | 510 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.cached` | 522 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.cached.decorator` | 539 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.cached.decorator.wrapper` | 541 |
| `argus/core/context_caching.py` | ⚙️ method | `ContextCache.get_or_set` | 566 |
| `argus/core/context_caching.py` | 🏛️ class | `ConversationCache` | 592 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.__init__` | 600 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.get_messages` | 609 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.add_message` | 613 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.add_user_message` | 638 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.add_assistant_message` | 647 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.add_system_message` | 656 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.get_recent_messages` | 665 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.clear_conversation` | 674 |
| `argus/core/context_caching.py` | ⚙️ method | `ConversationCache.summarize_and_truncate` | 678 |
| `argus/core/context_caching.py` | 🏛️ class | `EmbeddingCache` | 711 |
| `argus/core/context_caching.py` | ⚙️ method | `EmbeddingCache.__init__` | 719 |
| `argus/core/context_caching.py` | ⚙️ method | `EmbeddingCache._make_embedding_key` | 726 |
| `argus/core/context_caching.py` | ⚙️ method | `EmbeddingCache.get_embedding` | 731 |
| `argus/core/context_caching.py` | ⚙️ method | `EmbeddingCache.set_embedding` | 740 |
| `argus/core/context_caching.py` | ⚙️ method | `EmbeddingCache.get_or_compute` | 750 |
| `argus/core/context_caching.py` | ⚙️ method | `EmbeddingCache.batch_get_or_compute` | 767 |
| `argus/core/context_caching.py` | 🏛️ class | `LLMResponseCache` | 802 |
| `argus/core/context_caching.py` | ⚙️ method | `LLMResponseCache.__init__` | 810 |
| `argus/core/context_caching.py` | ⚙️ method | `LLMResponseCache._make_response_key` | 817 |
| `argus/core/context_caching.py` | ⚙️ method | `LLMResponseCache.get_response` | 844 |
| `argus/core/context_caching.py` | ⚙️ method | `LLMResponseCache.set_response` | 859 |
| `argus/core/context_caching.py` | ⚙️ method | `LLMResponseCache.cached_completion` | 875 |
| `argus/core/context_caching.py` | ⚙️ method | `LLMResponseCache.cached_completion.decorator` | 888 |
| `argus/core/context_caching.py` | ⚙️ method | `LLMResponseCache.cached_completion.decorator.wrapper` | 890 |
| `argus/core/context_caching.py` | 🔧 function | `create_cache` | 923 |
| `argus/core/context_compression.py` | 🏛️ class | `CompressionLevel` | 21 |
| `argus/core/context_compression.py` | 🏛️ class | `CompressionResult` | 30 |
| `argus/core/context_compression.py` | ⚙️ method | `CompressionResult.tokens_saved` | 41 |
| `argus/core/context_compression.py` | ⚙️ method | `CompressionResult.savings_percentage` | 46 |
| `argus/core/context_compression.py` | 🏛️ class | `Message` | 54 |
| `argus/core/context_compression.py` | 🏛️ class | `TokenCounter` | 62 |
| `argus/core/context_compression.py` | ⚙️ method | `TokenCounter.__init__` | 65 |
| `argus/core/context_compression.py` | ⚙️ method | `TokenCounter._get_encoder` | 69 |
| `argus/core/context_compression.py` | ⚙️ method | `TokenCounter.count` | 82 |
| `argus/core/context_compression.py` | ⚙️ method | `TokenCounter.count_messages` | 91 |
| `argus/core/context_compression.py` | 🏛️ class | `Compressor` | 104 |
| `argus/core/context_compression.py` | ⚙️ method | `Compressor.compress` | 110 |
| `argus/core/context_compression.py` | ⚙️ method | `Compressor.compress_with_result` | 119 |
| `argus/core/context_compression.py` | 🏛️ class | `WhitespaceCompressor` | 145 |
| `argus/core/context_compression.py` | ⚙️ method | `WhitespaceCompressor.compress` | 150 |
| `argus/core/context_compression.py` | 🏛️ class | `PunctuationCompressor` | 175 |
| `argus/core/context_compression.py` | ⚙️ method | `PunctuationCompressor.compress` | 180 |
| `argus/core/context_compression.py` | 🏛️ class | `StopwordCompressor` | 202 |
| `argus/core/context_compression.py` | ⚙️ method | `StopwordCompressor.compress` | 261 |
| `argus/core/context_compression.py` | 🏛️ class | `SentenceCompressor` | 298 |
| `argus/core/context_compression.py` | ⚙️ method | `SentenceCompressor.compress` | 342 |
| `argus/core/context_compression.py` | 🏛️ class | `CodeCompressor` | 403 |
| `argus/core/context_compression.py` | ⚙️ method | `CodeCompressor.compress` | 408 |
| `argus/core/context_compression.py` | 🏛️ class | `SemanticCompressor` | 453 |
| `argus/core/context_compression.py` | ⚙️ method | `SemanticCompressor.__init__` | 463 |
| `argus/core/context_compression.py` | ⚙️ method | `SemanticCompressor.compress` | 471 |
| `argus/core/context_compression.py` | 🏛️ class | `MessageCompressor` | 505 |
| `argus/core/context_compression.py` | ⚙️ method | `MessageCompressor.__init__` | 513 |
| `argus/core/context_compression.py` | ⚙️ method | `MessageCompressor.compress_message` | 527 |
| `argus/core/context_compression.py` | ⚙️ method | `MessageCompressor.compress_messages` | 545 |
| `argus/core/context_compression.py` | ⚙️ method | `MessageCompressor.sliding_window_compress` | 623 |
| `argus/core/context_compression.py` | 🏛️ class | `ContextCompressor` | 656 |
| `argus/core/context_compression.py` | ⚙️ method | `ContextCompressor.__init__` | 664 |
| `argus/core/context_compression.py` | ⚙️ method | `ContextCompressor.compress_text` | 690 |
| `argus/core/context_compression.py` | ⚙️ method | `ContextCompressor.compress_messages` | 769 |
| `argus/core/context_compression.py` | ⚙️ method | `ContextCompressor.auto_compress` | 780 |
| `argus/core/context_compression.py` | 🔧 function | `compress_text` | 823 |
| `argus/core/context_compression.py` | 🔧 function | `compress_to_tokens` | 844 |
| `argus/core/context_compression.py` | 🔧 function | `estimate_compression` | 866 |
| `argus/core/llm/anthropic.py` | 🏛️ class | `AnthropicLLM` | 24 |
| `argus/core/llm/anthropic.py` | ⚙️ method | `AnthropicLLM.__init__` | 64 |
| `argus/core/llm/anthropic.py` | ⚙️ method | `AnthropicLLM.provider_name` | 104 |
| `argus/core/llm/anthropic.py` | ⚙️ method | `AnthropicLLM._init_client` | 108 |
| `argus/core/llm/anthropic.py` | ⚙️ method | `AnthropicLLM.generate` | 130 |
| `argus/core/llm/anthropic.py` | ⚙️ method | `AnthropicLLM.stream` | 225 |
| `argus/core/llm/anthropic.py` | ⚙️ method | `AnthropicLLM._prepare_anthropic_messages` | 283 |
| `argus/core/llm/anthropic.py` | ⚙️ method | `AnthropicLLM.count_tokens` | 320 |
| `argus/core/llm/azure_openai.py` | 🏛️ class | `AzureOpenAILLM` | 18 |
| `argus/core/llm/azure_openai.py` | ⚙️ method | `AzureOpenAILLM.__init__` | 32 |
| `argus/core/llm/azure_openai.py` | ⚙️ method | `AzureOpenAILLM.provider_name` | 50 |
| `argus/core/llm/azure_openai.py` | ⚙️ method | `AzureOpenAILLM._init_client` | 53 |
| `argus/core/llm/azure_openai.py` | ⚙️ method | `AzureOpenAILLM.generate` | 66 |
| `argus/core/llm/azure_openai.py` | ⚙️ method | `AzureOpenAILLM.stream` | 92 |
| `argus/core/llm/azure_openai.py` | ⚙️ method | `AzureOpenAILLM.embed` | 108 |
| `argus/core/llm/base.py` | 🏛️ class | `MessageRole` | 22 |
| `argus/core/llm/base.py` | 🏛️ class | `Message` | 31 |
| `argus/core/llm/base.py` | ⚙️ method | `Message.system` | 67 |
| `argus/core/llm/base.py` | ⚙️ method | `Message.user` | 72 |
| `argus/core/llm/base.py` | ⚙️ method | `Message.assistant` | 77 |
| `argus/core/llm/base.py` | ⚙️ method | `Message.to_dict` | 81 |
| `argus/core/llm/base.py` | 🏛️ class | `LLMConfig` | 93 |
| `argus/core/llm/base.py` | 🏛️ class | `LLMUsage` | 173 |
| `argus/core/llm/base.py` | ⚙️ method | `LLMUsage.__add__` | 179 |
| `argus/core/llm/base.py` | 🏛️ class | `LLMResponse` | 189 |
| `argus/core/llm/base.py` | ⚙️ method | `LLMResponse.success` | 211 |
| `argus/core/llm/base.py` | 🏛️ class | `BaseLLM` | 216 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM.__init__` | 230 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM.provider_name` | 257 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM._init_client` | 262 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM.generate` | 267 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM.stream` | 294 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM.embed` | 320 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM.count_tokens` | 343 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM._prepare_messages` | 365 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM._measure_latency` | 395 |
| `argus/core/llm/base.py` | ⚙️ method | `BaseLLM.__repr__` | 399 |
| `argus/core/llm/base.py` | 🏛️ class | `EmbeddingModel` | 404 |
| `argus/core/llm/base.py` | ⚙️ method | `EmbeddingModel.__init__` | 417 |
| `argus/core/llm/base.py` | ⚙️ method | `EmbeddingModel._load_model` | 437 |
| `argus/core/llm/base.py` | ⚙️ method | `EmbeddingModel.dimension` | 461 |
| `argus/core/llm/base.py` | ⚙️ method | `EmbeddingModel.embed` | 466 |
| `argus/core/llm/base.py` | ⚙️ method | `EmbeddingModel.embed_query` | 501 |
| `argus/core/llm/base.py` | ⚙️ method | `EmbeddingModel.embed_documents` | 513 |
| `argus/core/llm/bedrock.py` | 🏛️ class | `BedrockLLM` | 19 |
| `argus/core/llm/bedrock.py` | ⚙️ method | `BedrockLLM.__init__` | 42 |
| `argus/core/llm/bedrock.py` | ⚙️ method | `BedrockLLM.provider_name` | 59 |
| `argus/core/llm/bedrock.py` | ⚙️ method | `BedrockLLM._init_client` | 62 |
| `argus/core/llm/bedrock.py` | ⚙️ method | `BedrockLLM._get_model_family` | 75 |
| `argus/core/llm/bedrock.py` | ⚙️ method | `BedrockLLM.generate` | 86 |
| `argus/core/llm/bedrock.py` | ⚙️ method | `BedrockLLM.stream` | 139 |
| `argus/core/llm/cerebras.py` | 🏛️ class | `CerebrasLLM` | 18 |
| `argus/core/llm/cerebras.py` | ⚙️ method | `CerebrasLLM.__init__` | 36 |
| `argus/core/llm/cerebras.py` | ⚙️ method | `CerebrasLLM.provider_name` | 49 |
| `argus/core/llm/cerebras.py` | ⚙️ method | `CerebrasLLM._init_client` | 52 |
| `argus/core/llm/cerebras.py` | ⚙️ method | `CerebrasLLM.generate` | 62 |
| `argus/core/llm/cerebras.py` | ⚙️ method | `CerebrasLLM.stream` | 88 |
| `argus/core/llm/cloudflare.py` | 🏛️ class | `CloudflareLLM` | 18 |
| `argus/core/llm/cloudflare.py` | ⚙️ method | `CloudflareLLM.__init__` | 35 |
| `argus/core/llm/cloudflare.py` | ⚙️ method | `CloudflareLLM.provider_name` | 50 |
| `argus/core/llm/cloudflare.py` | ⚙️ method | `CloudflareLLM._init_client` | 53 |
| `argus/core/llm/cloudflare.py` | ⚙️ method | `CloudflareLLM.generate` | 65 |
| `argus/core/llm/cloudflare.py` | ⚙️ method | `CloudflareLLM.stream` | 95 |
| `argus/core/llm/cohere.py` | 🏛️ class | `CohereLLM` | 24 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM.__init__` | 52 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM.provider_name` | 85 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM._init_client` | 89 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM.generate` | 107 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM.stream` | 199 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM._prepare_cohere_messages` | 253 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM.embed` | 302 |
| `argus/core/llm/cohere.py` | ⚙️ method | `CohereLLM.count_tokens` | 349 |
| `argus/core/llm/databricks.py` | 🏛️ class | `DatabricksLLM` | 18 |
| `argus/core/llm/databricks.py` | ⚙️ method | `DatabricksLLM.__init__` | 29 |
| `argus/core/llm/databricks.py` | ⚙️ method | `DatabricksLLM.provider_name` | 43 |
| `argus/core/llm/databricks.py` | ⚙️ method | `DatabricksLLM._init_client` | 46 |
| `argus/core/llm/databricks.py` | ⚙️ method | `DatabricksLLM.generate` | 58 |
| `argus/core/llm/databricks.py` | ⚙️ method | `DatabricksLLM.stream` | 84 |
| `argus/core/llm/deepseek.py` | 🏛️ class | `DeepSeekLLM` | 24 |
| `argus/core/llm/deepseek.py` | ⚙️ method | `DeepSeekLLM.__init__` | 50 |
| `argus/core/llm/deepseek.py` | ⚙️ method | `DeepSeekLLM.provider_name` | 67 |
| `argus/core/llm/deepseek.py` | ⚙️ method | `DeepSeekLLM._init_client` | 70 |
| `argus/core/llm/deepseek.py` | ⚙️ method | `DeepSeekLLM.generate` | 81 |
| `argus/core/llm/deepseek.py` | ⚙️ method | `DeepSeekLLM.stream` | 129 |
| `argus/core/llm/fireworks.py` | 🏛️ class | `FireworksLLM` | 18 |
| `argus/core/llm/fireworks.py` | ⚙️ method | `FireworksLLM.__init__` | 38 |
| `argus/core/llm/fireworks.py` | ⚙️ method | `FireworksLLM.provider_name` | 51 |
| `argus/core/llm/fireworks.py` | ⚙️ method | `FireworksLLM._init_client` | 54 |
| `argus/core/llm/fireworks.py` | ⚙️ method | `FireworksLLM.generate` | 64 |
| `argus/core/llm/fireworks.py` | ⚙️ method | `FireworksLLM.stream` | 90 |
| `argus/core/llm/gemini.py` | 🏛️ class | `GeminiLLM` | 24 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM.__init__` | 53 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM.provider_name` | 86 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM._init_client` | 90 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM.generate` | 111 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM.stream` | 199 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM._prepare_gemini_content` | 252 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM.embed` | 296 |
| `argus/core/llm/gemini.py` | ⚙️ method | `GeminiLLM.count_tokens` | 335 |
| `argus/core/llm/groq.py` | 🏛️ class | `GroqLLM` | 24 |
| `argus/core/llm/groq.py` | ⚙️ method | `GroqLLM.__init__` | 57 |
| `argus/core/llm/groq.py` | ⚙️ method | `GroqLLM.provider_name` | 90 |
| `argus/core/llm/groq.py` | ⚙️ method | `GroqLLM._init_client` | 94 |
| `argus/core/llm/groq.py` | ⚙️ method | `GroqLLM.generate` | 112 |
| `argus/core/llm/groq.py` | ⚙️ method | `GroqLLM.stream` | 201 |
| `argus/core/llm/groq.py` | ⚙️ method | `GroqLLM.transcribe` | 254 |
| `argus/core/llm/groq.py` | ⚙️ method | `GroqLLM.list_models` | 304 |
| `argus/core/llm/huggingface.py` | 🏛️ class | `HuggingFaceLLM` | 18 |
| `argus/core/llm/huggingface.py` | ⚙️ method | `HuggingFaceLLM.__init__` | 29 |
| `argus/core/llm/huggingface.py` | ⚙️ method | `HuggingFaceLLM.provider_name` | 43 |
| `argus/core/llm/huggingface.py` | ⚙️ method | `HuggingFaceLLM._init_client` | 46 |
| `argus/core/llm/huggingface.py` | ⚙️ method | `HuggingFaceLLM.generate` | 56 |
| `argus/core/llm/huggingface.py` | ⚙️ method | `HuggingFaceLLM.stream` | 84 |
| `argus/core/llm/huggingface.py` | ⚙️ method | `HuggingFaceLLM.embed` | 100 |
| `argus/core/llm/litellm.py` | 🏛️ class | `LiteLLMLLM` | 18 |
| `argus/core/llm/litellm.py` | ⚙️ method | `LiteLLMLLM.__init__` | 30 |
| `argus/core/llm/litellm.py` | ⚙️ method | `LiteLLMLLM.provider_name` | 44 |
| `argus/core/llm/litellm.py` | ⚙️ method | `LiteLLMLLM._init_client` | 47 |
| `argus/core/llm/litellm.py` | ⚙️ method | `LiteLLMLLM.generate` | 64 |
| `argus/core/llm/litellm.py` | ⚙️ method | `LiteLLMLLM.stream` | 95 |
| `argus/core/llm/litellm.py` | ⚙️ method | `LiteLLMLLM.embed` | 112 |
| `argus/core/llm/llamacpp.py` | 🏛️ class | `LlamaCppLLM` | 18 |
| `argus/core/llm/llamacpp.py` | ⚙️ method | `LlamaCppLLM.__init__` | 29 |
| `argus/core/llm/llamacpp.py` | ⚙️ method | `LlamaCppLLM.provider_name` | 49 |
| `argus/core/llm/llamacpp.py` | ⚙️ method | `LlamaCppLLM._init_client` | 52 |
| `argus/core/llm/llamacpp.py` | ⚙️ method | `LlamaCppLLM.generate` | 74 |
| `argus/core/llm/llamacpp.py` | ⚙️ method | `LlamaCppLLM.stream` | 108 |
| `argus/core/llm/mistral.py` | 🏛️ class | `MistralLLM` | 24 |
| `argus/core/llm/mistral.py` | ⚙️ method | `MistralLLM.__init__` | 58 |
| `argus/core/llm/mistral.py` | ⚙️ method | `MistralLLM.provider_name` | 91 |
| `argus/core/llm/mistral.py` | ⚙️ method | `MistralLLM._init_client` | 95 |
| `argus/core/llm/mistral.py` | ⚙️ method | `MistralLLM.generate` | 113 |
| `argus/core/llm/mistral.py` | ⚙️ method | `MistralLLM.stream` | 206 |
| `argus/core/llm/mistral.py` | ⚙️ method | `MistralLLM._prepare_mistral_messages` | 260 |
| `argus/core/llm/mistral.py` | ⚙️ method | `MistralLLM.embed` | 303 |
| `argus/core/llm/nvidia.py` | 🏛️ class | `NvidiaLLM` | 18 |
| `argus/core/llm/nvidia.py` | ⚙️ method | `NvidiaLLM.__init__` | 31 |
| `argus/core/llm/nvidia.py` | ⚙️ method | `NvidiaLLM.provider_name` | 45 |
| `argus/core/llm/nvidia.py` | ⚙️ method | `NvidiaLLM._init_client` | 48 |
| `argus/core/llm/nvidia.py` | ⚙️ method | `NvidiaLLM.generate` | 58 |
| `argus/core/llm/nvidia.py` | ⚙️ method | `NvidiaLLM.stream` | 84 |
| `argus/core/llm/ollama.py` | 🏛️ class | `OllamaLLM` | 27 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM.__init__` | 58 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM.provider_name` | 96 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM._init_client` | 100 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM.generate` | 122 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM.stream` | 220 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM._prepare_ollama_messages` | 292 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM.embed` | 324 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM.list_models` | 367 |
| `argus/core/llm/ollama.py` | ⚙️ method | `OllamaLLM.pull_model` | 383 |
| `argus/core/llm/openai.py` | 🏛️ class | `OpenAILLM` | 24 |
| `argus/core/llm/openai.py` | ⚙️ method | `OpenAILLM.__init__` | 60 |
| `argus/core/llm/openai.py` | ⚙️ method | `OpenAILLM.provider_name` | 106 |
| `argus/core/llm/openai.py` | ⚙️ method | `OpenAILLM._init_client` | 110 |
| `argus/core/llm/openai.py` | ⚙️ method | `OpenAILLM.generate` | 138 |
| `argus/core/llm/openai.py` | ⚙️ method | `OpenAILLM.stream` | 235 |
| `argus/core/llm/openai.py` | ⚙️ method | `OpenAILLM.embed` | 288 |
| `argus/core/llm/openai.py` | ⚙️ method | `OpenAILLM.count_tokens` | 327 |
| `argus/core/llm/perplexity.py` | 🏛️ class | `PerplexityLLM` | 18 |
| `argus/core/llm/perplexity.py` | ⚙️ method | `PerplexityLLM.__init__` | 40 |
| `argus/core/llm/perplexity.py` | ⚙️ method | `PerplexityLLM.provider_name` | 53 |
| `argus/core/llm/perplexity.py` | ⚙️ method | `PerplexityLLM._init_client` | 56 |
| `argus/core/llm/perplexity.py` | ⚙️ method | `PerplexityLLM.generate` | 66 |
| `argus/core/llm/perplexity.py` | ⚙️ method | `PerplexityLLM.stream` | 93 |
| `argus/core/llm/registry.py` | 🏛️ class | `LLMRegistry` | 19 |
| `argus/core/llm/registry.py` | ⚙️ method | `LLMRegistry.register` | 43 |
| `argus/core/llm/registry.py` | ⚙️ method | `LLMRegistry.get` | 55 |
| `argus/core/llm/registry.py` | ⚙️ method | `LLMRegistry.list_providers` | 137 |
| `argus/core/llm/registry.py` | ⚙️ method | `LLMRegistry.has_provider` | 147 |
| `argus/core/llm/registry.py` | ⚙️ method | `LLMRegistry.clear_cache` | 160 |
| `argus/core/llm/registry.py` | 🔧 function | `_register_default_providers` | 166 |
| `argus/core/llm/registry.py` | 🔧 function | `get_llm` | 254 |
| `argus/core/llm/registry.py` | 🔧 function | `register_provider` | 288 |
| `argus/core/llm/registry.py` | 🔧 function | `list_providers` | 308 |
| `argus/core/llm/replicate.py` | 🏛️ class | `ReplicateLLM` | 18 |
| `argus/core/llm/replicate.py` | ⚙️ method | `ReplicateLLM.__init__` | 35 |
| `argus/core/llm/replicate.py` | ⚙️ method | `ReplicateLLM.provider_name` | 48 |
| `argus/core/llm/replicate.py` | ⚙️ method | `ReplicateLLM._init_client` | 51 |
| `argus/core/llm/replicate.py` | ⚙️ method | `ReplicateLLM.generate` | 62 |
| `argus/core/llm/replicate.py` | ⚙️ method | `ReplicateLLM.stream` | 90 |
| `argus/core/llm/sambanova.py` | 🏛️ class | `SambanovaLLM` | 18 |
| `argus/core/llm/sambanova.py` | ⚙️ method | `SambanovaLLM.__init__` | 37 |
| `argus/core/llm/sambanova.py` | ⚙️ method | `SambanovaLLM.provider_name` | 50 |
| `argus/core/llm/sambanova.py` | ⚙️ method | `SambanovaLLM._init_client` | 53 |
| `argus/core/llm/sambanova.py` | ⚙️ method | `SambanovaLLM.generate` | 63 |
| `argus/core/llm/sambanova.py` | ⚙️ method | `SambanovaLLM.stream` | 89 |
| `argus/core/llm/snowflake.py` | 🏛️ class | `SnowflakeLLM` | 18 |
| `argus/core/llm/snowflake.py` | ⚙️ method | `SnowflakeLLM.__init__` | 35 |
| `argus/core/llm/snowflake.py` | ⚙️ method | `SnowflakeLLM.provider_name` | 60 |
| `argus/core/llm/snowflake.py` | ⚙️ method | `SnowflakeLLM._init_client` | 63 |
| `argus/core/llm/snowflake.py` | ⚙️ method | `SnowflakeLLM.generate` | 79 |
| `argus/core/llm/snowflake.py` | ⚙️ method | `SnowflakeLLM.stream` | 117 |
| `argus/core/llm/together.py` | 🏛️ class | `TogetherLLM` | 18 |
| `argus/core/llm/together.py` | ⚙️ method | `TogetherLLM.__init__` | 38 |
| `argus/core/llm/together.py` | ⚙️ method | `TogetherLLM.provider_name` | 51 |
| `argus/core/llm/together.py` | ⚙️ method | `TogetherLLM._init_client` | 54 |
| `argus/core/llm/together.py` | ⚙️ method | `TogetherLLM.generate` | 64 |
| `argus/core/llm/together.py` | ⚙️ method | `TogetherLLM.stream` | 90 |
| `argus/core/llm/together.py` | ⚙️ method | `TogetherLLM.embed` | 106 |
| `argus/core/llm/vertex.py` | 🏛️ class | `VertexAILLM` | 18 |
| `argus/core/llm/vertex.py` | ⚙️ method | `VertexAILLM.__init__` | 34 |
| `argus/core/llm/vertex.py` | ⚙️ method | `VertexAILLM.provider_name` | 51 |
| `argus/core/llm/vertex.py` | ⚙️ method | `VertexAILLM._init_client` | 54 |
| `argus/core/llm/vertex.py` | ⚙️ method | `VertexAILLM.generate` | 66 |
| `argus/core/llm/vertex.py` | ⚙️ method | `VertexAILLM.stream` | 102 |
| `argus/core/llm/vllm.py` | 🏛️ class | `VllmLLM` | 18 |
| `argus/core/llm/vllm.py` | ⚙️ method | `VllmLLM.__init__` | 29 |
| `argus/core/llm/vllm.py` | ⚙️ method | `VllmLLM.provider_name` | 43 |
| `argus/core/llm/vllm.py` | ⚙️ method | `VllmLLM._init_client` | 46 |
| `argus/core/llm/vllm.py` | ⚙️ method | `VllmLLM.generate` | 56 |
| `argus/core/llm/vllm.py` | ⚙️ method | `VllmLLM.stream` | 82 |
| `argus/core/llm/watsonx.py` | 🏛️ class | `WatsonxLLM` | 18 |
| `argus/core/llm/watsonx.py` | ⚙️ method | `WatsonxLLM.__init__` | 34 |
| `argus/core/llm/watsonx.py` | ⚙️ method | `WatsonxLLM.provider_name` | 51 |
| `argus/core/llm/watsonx.py` | ⚙️ method | `WatsonxLLM._init_client` | 54 |
| `argus/core/llm/watsonx.py` | ⚙️ method | `WatsonxLLM.generate` | 68 |
| `argus/core/llm/watsonx.py` | ⚙️ method | `WatsonxLLM.stream` | 104 |
| `argus/core/llm/xai.py` | 🏛️ class | `XaiLLM` | 18 |
| `argus/core/llm/xai.py` | ⚙️ method | `XaiLLM.__init__` | 31 |
| `argus/core/llm/xai.py` | ⚙️ method | `XaiLLM.provider_name` | 43 |
| `argus/core/llm/xai.py` | ⚙️ method | `XaiLLM._init_client` | 46 |
| `argus/core/llm/xai.py` | ⚙️ method | `XaiLLM.generate` | 56 |
| `argus/core/llm/xai.py` | ⚙️ method | `XaiLLM.stream` | 83 |
| `argus/core/models.py` | 🔧 function | `generate_uuid` | 26 |
| `argus/core/models.py` | 🔧 function | `compute_hash` | 31 |
| `argus/core/models.py` | 🏛️ class | `SourceType` | 47 |
| `argus/core/models.py` | 🏛️ class | `Document` | 58 |
| `argus/core/models.py` | ⚙️ method | `Document.content_hash` | 134 |
| `argus/core/models.py` | ⚙️ method | `Document.word_count` | 140 |
| `argus/core/models.py` | ⚙️ method | `Document.__hash__` | 144 |
| `argus/core/models.py` | ⚙️ method | `Document.__eq__` | 148 |
| `argus/core/models.py` | 🏛️ class | `Chunk` | 155 |
| `argus/core/models.py` | ⚙️ method | `Chunk.span` | 224 |
| `argus/core/models.py` | ⚙️ method | `Chunk.length` | 230 |
| `argus/core/models.py` | ⚙️ method | `Chunk.__hash__` | 234 |
| `argus/core/models.py` | ⚙️ method | `Chunk.__eq__` | 238 |
| `argus/core/models.py` | 🏛️ class | `Embedding` | 245 |
| `argus/core/models.py` | ⚙️ method | `Embedding.dimension` | 288 |
| `argus/core/models.py` | ⚙️ method | `Embedding.__hash__` | 292 |
| `argus/core/models.py` | 🏛️ class | `CitationType` | 297 |
| `argus/core/models.py` | 🏛️ class | `Citation` | 305 |
| `argus/core/models.py` | 🏛️ class | `Claim` | 354 |
| `argus/core/models.py` | ⚙️ method | `Claim.validate_confidence` | 414 |
| `argus/core/models.py` | ⚙️ method | `Claim.__hash__` | 418 |
| `argus/core/models.py` | 🏛️ class | `NodeType` | 423 |
| `argus/core/models.py` | 🏛️ class | `NodeBase` | 432 |
| `argus/core/models.py` | ⚙️ method | `NodeBase.__hash__` | 488 |
| `argus/core/models.py` | ⚙️ method | `NodeBase.__eq__` | 492 |
| `argus/core/models.py` | ⚙️ method | `NodeBase.to_prov_dict` | 498 |
| `argus/core/models.py` | 🏛️ class | `ScoredItem` | 515 |
| `argus/core/models.py` | 🏛️ class | `RetrievalResult` | 553 |
| `argus/core/openapi.py` | 🏛️ class | `AuthType` | 22 |
| `argus/core/openapi.py` | 🏛️ class | `SecurityScheme` | 32 |
| `argus/core/openapi.py` | 🏛️ class | `APIParameter` | 43 |
| `argus/core/openapi.py` | ⚙️ method | `APIParameter.param_type` | 53 |
| `argus/core/openapi.py` | 🏛️ class | `APIOperation` | 58 |
| `argus/core/openapi.py` | 🏛️ class | `APISpec` | 74 |
| `argus/core/openapi.py` | 🏛️ class | `OpenAPIParser` | 86 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIParser.__init__` | 89 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIParser.parse` | 92 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIParser._parse_security_scheme` | 159 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIParser._parse_parameters` | 187 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIParser._parse_operation` | 207 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIParser._resolve_ref` | 268 |
| `argus/core/openapi.py` | 🏛️ class | `OpenAPIClient` | 285 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.__init__` | 297 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.from_dict` | 315 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.from_json` | 328 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.from_file` | 340 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.from_url` | 363 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient._get_session` | 389 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient._configure_auth` | 400 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient._setup_methods` | 432 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient._to_snake_case` | 444 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient._create_method` | 450 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient._create_method.method` | 452 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.execute` | 474 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient._execute_operation` | 492 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.list_operations` | 571 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIClient.get_operation_schema` | 587 |
| `argus/core/openapi.py` | 🏛️ class | `OpenAPIToolGenerator` | 620 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIToolGenerator.__init__` | 627 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIToolGenerator.from_url` | 632 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIToolGenerator.from_file` | 645 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIToolGenerator.generate_tool_class` | 661 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIToolGenerator._generate_action_method` | 800 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIToolGenerator._to_snake_case` | 879 |
| `argus/core/openapi.py` | ⚙️ method | `OpenAPIToolGenerator._to_env_var` | 884 |
| `argus/core/openapi.py` | 🔧 function | `load_openapi_spec` | 891 |
| `argus/core/openapi.py` | 🔧 function | `create_client` | 922 |
| `argus/core/openapi.py` | 🔧 function | `generate_tool_code` | 942 |
| `argus/crux/agent_card.py` | 🏛️ class | `AgentCalibration` | 35 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCalibration.is_well_calibrated` | 84 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCalibration.is_credible` | 90 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCalibration.effective_weight` | 96 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCalibration.neutral` | 106 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCalibration.from_metrics` | 117 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCalibration.update` | 140 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCalibration.to_dict` | 177 |
| `argus/crux/agent_card.py` | 🏛️ class | `AgentCapabilities` | 191 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCapabilities.specialist` | 233 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCapabilities.refuter` | 244 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCapabilities.jury` | 255 |
| `argus/crux/agent_card.py` | ⚙️ method | `AgentCapabilities.to_dict` | 265 |
| `argus/crux/agent_card.py` | 🏛️ class | `EpistemicAgentCard` | 276 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.is_expired` | 370 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.card_hash` | 378 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.from_argus_agent` | 390 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.can_handle_domain` | 438 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.domain_match_score` | 461 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.refresh_calibration` | 484 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.to_dict` | 512 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.to_json` | 532 |
| `argus/crux/agent_card.py` | ⚙️ method | `EpistemicAgentCard.from_json` | 537 |
| `argus/crux/agent_card.py` | 🏛️ class | `EACRegistry` | 576 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.__init__` | 584 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.register` | 590 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.unregister` | 612 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.get` | 639 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.get_card` | 643 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.get_by_type` | 647 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.get_by_domain` | 652 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.get_challengers` | 657 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.get_all` | 688 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.__len__` | 692 |
| `argus/crux/agent_card.py` | ⚙️ method | `EACRegistry.__contains__` | 696 |
| `argus/crux/auction.py` | 🏛️ class | `BidRejectionReason` | 41 |
| `argus/crux/auction.py` | 🏛️ class | `BidResult` | 52 |
| `argus/crux/auction.py` | ⚙️ method | `BidResult.to_dict` | 65 |
| `argus/crux/auction.py` | 🏛️ class | `AuctionResult` | 75 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionResult.__post_init__` | 102 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionResult.num_bids` | 107 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionResult.duration_seconds` | 112 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionResult.to_dict` | 118 |
| `argus/crux/auction.py` | 🏛️ class | `BidEvaluator` | 135 |
| `argus/crux/auction.py` | ⚙️ method | `BidEvaluator.__init__` | 152 |
| `argus/crux/auction.py` | ⚙️ method | `BidEvaluator.validate_bid` | 167 |
| `argus/crux/auction.py` | ⚙️ method | `BidEvaluator.compute_bid_score` | 205 |
| `argus/crux/auction.py` | ⚙️ method | `BidEvaluator.rank_bids` | 249 |
| `argus/crux/auction.py` | ⚙️ method | `BidEvaluator.select_winner` | 281 |
| `argus/crux/auction.py` | 🏛️ class | `ChallengerAuction` | 301 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.__init__` | 322 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.start` | 351 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.get_session` | 390 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.get_session_for_cb` | 394 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.submit_bid` | 401 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.close` | 434 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.close_if_timeout` | 477 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.run_auction` | 496 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.get_result` | 533 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.on_bid_received` | 537 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.on_auction_complete` | 541 |
| `argus/crux/auction.py` | ⚙️ method | `ChallengerAuction.active_auctions` | 546 |
| `argus/crux/auction.py` | 🏛️ class | `AuctionSession` | 551 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.__init__` | 558 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.cb_id` | 591 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.state` | 596 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.is_expired` | 601 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.time_remaining` | 606 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.num_bids` | 612 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.submit_bid` | 616 |
| `argus/crux/auction.py` | ⚙️ method | `AuctionSession.close` | 667 |
| `argus/crux/auction.py` | 🔧 function | `create_bid` | 731 |
| `argus/crux/brp.py` | 🏛️ class | `ContradictionType` | 43 |
| `argus/crux/brp.py` | 🏛️ class | `Contradiction` | 51 |
| `argus/crux/brp.py` | ⚙️ method | `Contradiction.proposition_id` | 69 |
| `argus/crux/brp.py` | ⚙️ method | `Contradiction.to_dict` | 73 |
| `argus/crux/brp.py` | 🏛️ class | `MiniDebateRound` | 85 |
| `argus/crux/brp.py` | ⚙️ method | `MiniDebateRound.to_dict` | 102 |
| `argus/crux/brp.py` | 🏛️ class | `BRPSession` | 116 |
| `argus/crux/brp.py` | ⚙️ method | `BRPSession.is_active` | 147 |
| `argus/crux/brp.py` | ⚙️ method | `BRPSession.current_round` | 152 |
| `argus/crux/brp.py` | ⚙️ method | `BRPSession.transition` | 156 |
| `argus/crux/brp.py` | ⚙️ method | `BRPSession.add_round` | 171 |
| `argus/crux/brp.py` | ⚙️ method | `BRPSession.duration_seconds` | 176 |
| `argus/crux/brp.py` | ⚙️ method | `BRPSession.to_dict` | 183 |
| `argus/crux/brp.py` | 🏛️ class | `ContradictionDetector` | 210 |
| `argus/crux/brp.py` | ⚙️ method | `ContradictionDetector.__init__` | 218 |
| `argus/crux/brp.py` | ⚙️ method | `ContradictionDetector.detect` | 227 |
| `argus/crux/brp.py` | ⚙️ method | `ContradictionDetector.find_contradictions` | 277 |
| `argus/crux/brp.py` | 🏛️ class | `BayesianMerger` | 311 |
| `argus/crux/brp.py` | ⚙️ method | `BayesianMerger.compute_merged_posterior` | 321 |
| `argus/crux/brp.py` | ⚙️ method | `BayesianMerger.merge_distributions` | 349 |
| `argus/crux/brp.py` | 🏛️ class | `ReconciliationResult` | 393 |
| `argus/crux/brp.py` | ⚙️ method | `ReconciliationResult.to_dict` | 408 |
| `argus/crux/brp.py` | 🏛️ class | `BeliefReconciliationProtocol` | 423 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.__init__` | 440 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.detect_contradiction` | 466 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.find_all_contradictions` | 483 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.start_session` | 490 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.run_mini_debate` | 522 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.perform_merge` | 554 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.complete_session` | 606 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.reconcile` | 671 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.get_session` | 749 |
| `argus/crux/brp.py` | ⚙️ method | `BeliefReconciliationProtocol.get_history` | 753 |
| `argus/crux/claim_bundle.py` | 🏛️ class | `ClaimBundle` | 42 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.__init__` | 193 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle._compute_cb_id` | 214 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle._compute_lineage_hash` | 227 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.is_expired` | 238 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.challenge_expired` | 245 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.is_supporting` | 253 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.is_attacking` | 259 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.effective_weight` | 265 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.is_merged` | 277 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.close_challenge` | 281 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.mark_unchallenged` | 285 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.conflicts_with` | 291 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.verify_signature` | 321 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle._get_signable_content` | 353 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.to_dict` | 365 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.to_json` | 391 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundle.from_dict` | 396 |
| `argus/crux/claim_bundle.py` | 🏛️ class | `ClaimBundleFactory` | 437 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundleFactory.from_evidence` | 446 |
| `argus/crux/claim_bundle.py` | ⚙️ method | `ClaimBundleFactory.from_proposition_posterior` | 504 |
| `argus/crux/claim_bundle.py` | 🔧 function | `merge_claim_bundles` | 551 |
| `argus/crux/edr.py` | 🏛️ class | `EDRCheckpoint` | 43 |
| `argus/crux/edr.py` | ⚙️ method | `EDRCheckpoint.__post_init__` | 68 |
| `argus/crux/edr.py` | ⚙️ method | `EDRCheckpoint._compute_hash` | 72 |
| `argus/crux/edr.py` | ⚙️ method | `EDRCheckpoint.to_dict` | 82 |
| `argus/crux/edr.py` | ⚙️ method | `EDRCheckpoint.from_dict` | 95 |
| `argus/crux/edr.py` | 🏛️ class | `BeliefConflict` | 110 |
| `argus/crux/edr.py` | ⚙️ method | `BeliefConflict.__post_init__` | 127 |
| `argus/crux/edr.py` | ⚙️ method | `BeliefConflict.to_dict` | 130 |
| `argus/crux/edr.py` | 🏛️ class | `SyncResult` | 142 |
| `argus/crux/edr.py` | ⚙️ method | `SyncResult.to_dict` | 161 |
| `argus/crux/edr.py` | 🏛️ class | `EDRSynchronizer` | 173 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.__init__` | 190 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.create_checkpoint` | 217 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.find_checkpoint` | 261 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.get_latest_checkpoint` | 280 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.register_bundle` | 290 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.compute_delta` | 298 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.detect_conflicts` | 367 |
| `argus/crux/edr.py` | ⚙️ method | `EDRSynchronizer.sync` | 408 |
| `argus/crux/edr.py` | 🏛️ class | `EpistemicDeadReckoning` | 492 |
| `argus/crux/edr.py` | ⚙️ method | `EpistemicDeadReckoning.__init__` | 512 |
| `argus/crux/edr.py` | ⚙️ method | `EpistemicDeadReckoning.enabled` | 531 |
| `argus/crux/edr.py` | ⚙️ method | `EpistemicDeadReckoning.checkpoint` | 535 |
| `argus/crux/edr.py` | ⚙️ method | `EpistemicDeadReckoning.reconnect` | 555 |
| `argus/crux/edr.py` | ⚙️ method | `EpistemicDeadReckoning.register_bundle` | 575 |
| `argus/crux/edr.py` | ⚙️ method | `EpistemicDeadReckoning.get_checkpoint` | 579 |
| `argus/crux/ledger.py` | 🏛️ class | `CredibilityEntry` | 39 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityEntry.compute_hash` | 70 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityEntry.to_dict` | 85 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityEntry.from_dict` | 101 |
| `argus/crux/ledger.py` | 🏛️ class | `CredibilityUpdate` | 119 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityUpdate.delta` | 139 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityUpdate.to_dict` | 143 |
| `argus/crux/ledger.py` | 🏛️ class | `AgentCredibilityState` | 157 |
| `argus/crux/ledger.py` | ⚙️ method | `AgentCredibilityState.average_brier` | 179 |
| `argus/crux/ledger.py` | ⚙️ method | `AgentCredibilityState.to_dict` | 185 |
| `argus/crux/ledger.py` | 🏛️ class | `CredibilityLedger` | 201 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.__init__` | 232 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.head_hash` | 268 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.chain_length` | 275 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger._get_or_create_state` | 279 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.record_prediction` | 285 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.record_outcome` | 327 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.record_prediction_with_outcome` | 378 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger._update_credibility` | 428 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger._check_suspension` | 468 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.get_credibility` | 488 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.get_agent_state` | 503 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.is_suspended` | 507 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.reinstate_agent` | 512 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.get_entries_for_agent` | 531 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.get_entries_since` | 554 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.get_entries_after_hash` | 561 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.verify_chain_integrity` | 577 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.detect_sybil_agents` | 600 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger._append_to_file` | 636 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger._load` | 641 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.get_all_agent_ids` | 672 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.get_credibility_history` | 676 |
| `argus/crux/ledger.py` | ⚙️ method | `CredibilityLedger.export_statistics` | 711 |
| `argus/crux/models.py` | 🔧 function | `generate_crux_id` | 26 |
| `argus/crux/models.py` | 🔧 function | `compute_crux_hash` | 31 |
| `argus/crux/models.py` | 🏛️ class | `CRUXVersion` | 38 |
| `argus/crux/models.py` | 🏛️ class | `Polarity` | 43 |
| `argus/crux/models.py` | 🏛️ class | `DistributionType` | 50 |
| `argus/crux/models.py` | 🏛️ class | `BRPState` | 57 |
| `argus/crux/models.py` | 🏛️ class | `AuctionState` | 67 |
| `argus/crux/models.py` | 🏛️ class | `BetaDistribution` | 76 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.mean` | 107 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.mode` | 113 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.variance` | 121 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.std` | 128 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.concentration` | 134 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.pdf` | 138 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.cdf` | 161 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.sample` | 174 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.quantile` | 187 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.credible_interval` | 200 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.from_mean_concentration` | 214 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.from_mean_std` | 234 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.uniform` | 261 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.jeffreys` | 266 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.update` | 270 |
| `argus/crux/models.py` | ⚙️ method | `BetaDistribution.to_dict` | 290 |
| `argus/crux/models.py` | 🏛️ class | `ConfidenceDistribution` | 301 |
| `argus/crux/models.py` | ⚙️ method | `ConfidenceDistribution.from_beta` | 339 |
| `argus/crux/models.py` | ⚙️ method | `ConfidenceDistribution.from_point` | 354 |
| `argus/crux/models.py` | ⚙️ method | `ConfidenceDistribution.from_mean_std` | 363 |
| `argus/crux/models.py` | ⚙️ method | `ConfidenceDistribution.sample` | 377 |
| `argus/crux/models.py` | ⚙️ method | `ConfidenceDistribution.credible_interval` | 388 |
| `argus/crux/models.py` | ⚙️ method | `ConfidenceDistribution.to_dict` | 401 |
| `argus/crux/models.py` | 🏛️ class | `ChallengerBid` | 415 |
| `argus/crux/models.py` | ⚙️ method | `ChallengerBid.__post_init__` | 438 |
| `argus/crux/models.py` | ⚙️ method | `ChallengerBid.is_expired` | 443 |
| `argus/crux/models.py` | ⚙️ method | `ChallengerBid.to_dict` | 447 |
| `argus/crux/models.py` | 🏛️ class | `BRPResolution` | 462 |
| `argus/crux/models.py` | ⚙️ method | `BRPResolution.to_dict` | 489 |
| `argus/crux/models.py` | 🏛️ class | `EDRDelta` | 509 |
| `argus/crux/models.py` | ⚙️ method | `EDRDelta.num_changes` | 533 |
| `argus/crux/models.py` | ⚙️ method | `EDRDelta.to_dict` | 541 |
| `argus/crux/models.py` | 🏛️ class | `CRUXConfig` | 554 |
| `argus/crux/models.py` | ⚙️ method | `CRUXConfig.validate_dfs_weights` | 654 |
| `argus/crux/models.py` | ⚙️ method | `CRUXConfig.to_dict` | 664 |
| `argus/crux/orchestrator.py` | 🏛️ class | `CRUXSessionState` | 80 |
| `argus/crux/orchestrator.py` | 🏛️ class | `CRUXSessionStats` | 92 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXSessionStats.to_dict` | 113 |
| `argus/crux/orchestrator.py` | 🏛️ class | `CRUXSession` | 127 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXSession.duration_seconds` | 167 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXSession.add_claim_bundle` | 172 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXSession.add_auction` | 182 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXSession.add_brp_session` | 192 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXSession.get_claim_bundles_for_proposition` | 197 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXSession.to_dict` | 202 |
| `argus/crux/orchestrator.py` | 🏛️ class | `CRUXDebateResult` | 220 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXDebateResult.was_challenged` | 248 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXDebateResult.credibility_impact` | 253 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXDebateResult.to_dict` | 260 |
| `argus/crux/orchestrator.py` | 🏛️ class | `CRUXOrchestrator` | 281 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.__init__` | 307 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._setup_agent_cards` | 379 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.debate` | 441 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._update_agent_domains` | 584 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._convert_to_claim_bundles` | 604 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._compute_evidence_posterior` | 705 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._run_auction` | 736 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._get_potential_challengers` | 778 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._reconcile_claims` | 811 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._find_contradictions` | 875 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator._update_credibility` | 908 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.on_claim_bundle` | 947 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.on_brp_triggered` | 951 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.on_auction_complete` | 955 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.current_session` | 961 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.session_history` | 966 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.register_agent` | 971 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.get_agent_card` | 976 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.get_card_endpoint` | 981 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.submit_claim_endpoint` | 988 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.submit_bid_endpoint` | 1002 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.get_ledger_endpoint` | 1024 |
| `argus/crux/orchestrator.py` | ⚙️ method | `CRUXOrchestrator.sync_endpoint` | 1036 |
| `argus/crux/routing.py` | 🏛️ class | `DialecticalFitnessScore` | 35 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalFitnessScore.to_dict` | 61 |
| `argus/crux/routing.py` | 🏛️ class | `DomainMatcher` | 77 |
| `argus/crux/routing.py` | ⚙️ method | `DomainMatcher.extract_domains_from_text` | 106 |
| `argus/crux/routing.py` | ⚙️ method | `DomainMatcher.compute_domain_match` | 126 |
| `argus/crux/routing.py` | 🏛️ class | `AdversarialAssessor` | 177 |
| `argus/crux/routing.py` | ⚙️ method | `AdversarialAssessor.compute_adversarial_potential` | 186 |
| `argus/crux/routing.py` | 🏛️ class | `RecencyTracker` | 227 |
| `argus/crux/routing.py` | ⚙️ method | `RecencyTracker.__init__` | 235 |
| `argus/crux/routing.py` | ⚙️ method | `RecencyTracker.record_engagement` | 245 |
| `argus/crux/routing.py` | ⚙️ method | `RecencyTracker.compute_recency_penalty` | 259 |
| `argus/crux/routing.py` | ⚙️ method | `RecencyTracker.cleanup_old` | 289 |
| `argus/crux/routing.py` | 🔧 function | `compute_dfs` | 315 |
| `argus/crux/routing.py` | 🏛️ class | `DialecticalRouter` | 377 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.__init__` | 390 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.set_agent_prior` | 409 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.get_agent_prior` | 425 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.rank_challengers` | 438 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.get_best_challenger` | 491 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.record_challenge` | 507 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.get_routing_table` | 519 |
| `argus/crux/routing.py` | ⚙️ method | `DialecticalRouter.visualize_routing` | 535 |
| `argus/crux/visualization.py` | 🔧 function | `_check_plotly` | 97 |
| `argus/crux/visualization.py` | 🔧 function | `_apply_dark_layout` | 106 |
| `argus/crux/visualization.py` | 🔧 function | `_get_polarity_color` | 115 |
| `argus/crux/visualization.py` | 🔧 function | `_get_credibility_color` | 124 |
| `argus/crux/visualization.py` | 🔧 function | `plot_crux_debate_flow` | 137 |
| `argus/crux/visualization.py` | 🔧 function | `plot_credibility_evolution` | 419 |
| `argus/crux/visualization.py` | 🔧 function | `plot_brp_merge` | 524 |
| `argus/crux/visualization.py` | 🔧 function | `plot_dfs_heatmap` | 655 |
| `argus/crux/visualization.py` | 🔧 function | `plot_dfs_breakdown` | 734 |
| `argus/crux/visualization.py` | 🔧 function | `plot_auction_results` | 792 |
| `argus/crux/visualization.py` | 🔧 function | `plot_auction_timeline` | 894 |
| `argus/crux/visualization.py` | 🔧 function | `create_crux_dashboard` | 971 |
| `argus/crux/visualization.py` | 🔧 function | `export_debate_static` | 1084 |
| `argus/crux/visualization.py` | 🔧 function | `export_session_json` | 1139 |
| `argus/crux/visualization.py` | 🔧 function | `build_dfs_matrix` | 1176 |
| `argus/debate/visualization.py` | 🏛️ class | `ArgumentType` | 99 |
| `argus/debate/visualization.py` | 🏛️ class | `ArgumentStatus` | 109 |
| `argus/debate/visualization.py` | 🏛️ class | `Argument` | 119 |
| `argus/debate/visualization.py` | 🏛️ class | `DebateRound` | 134 |
| `argus/debate/visualization.py` | 🏛️ class | `DebateSession` | 146 |
| `argus/debate/visualization.py` | ⚙️ method | `DebateSession.all_arguments` | 158 |
| `argus/debate/visualization.py` | ⚙️ method | `DebateSession.duration_seconds` | 166 |
| `argus/debate/visualization.py` | 🔧 function | `_check_plotly` | 172 |
| `argus/debate/visualization.py` | 🔧 function | `_check_networkx` | 181 |
| `argus/debate/visualization.py` | 🔧 function | `plot_argument_flow` | 194 |
| `argus/debate/visualization.py` | 🔧 function | `_hierarchical_layout` | 387 |
| `argus/debate/visualization.py` | 🔧 function | `plot_debate_timeline` | 430 |
| `argus/debate/visualization.py` | 🔧 function | `plot_agent_performance` | 543 |
| `argus/debate/visualization.py` | 🔧 function | `plot_confidence_evolution` | 707 |
| `argus/debate/visualization.py` | 🔧 function | `plot_round_summary` | 813 |
| `argus/debate/visualization.py` | 🔧 function | `plot_interaction_heatmap` | 938 |
| `argus/debate/visualization.py` | 🔧 function | `plot_argument_type_distribution` | 1005 |
| `argus/debate/visualization.py` | 🔧 function | `create_debate_dashboard` | 1065 |
| `argus/debate/visualization.py` | 🔧 function | `export_debate_html` | 1267 |
| `argus/debate/visualization.py` | 🔧 function | `export_debate_png` | 1295 |
| `argus/debate/visualization.py` | 🔧 function | `generate_debate_report` | 1321 |
| `argus/decision/bayesian.py` | 🔧 function | `log_odds` | 27 |
| `argus/decision/bayesian.py` | 🔧 function | `from_log_odds` | 44 |
| `argus/decision/bayesian.py` | 🔧 function | `compute_likelihood_ratio` | 61 |
| `argus/decision/bayesian.py` | 🏛️ class | `UpdateResult` | 101 |
| `argus/decision/bayesian.py` | 🏛️ class | `BayesianUpdater` | 113 |
| `argus/decision/bayesian.py` | ⚙️ method | `BayesianUpdater.__init__` | 129 |
| `argus/decision/bayesian.py` | ⚙️ method | `BayesianUpdater.update` | 144 |
| `argus/decision/bayesian.py` | ⚙️ method | `BayesianUpdater.update_from_graph` | 190 |
| `argus/decision/bayesian.py` | 🔧 function | `update_posterior` | 269 |
| `argus/decision/bayesian.py` | 🔧 function | `batch_update` | 311 |
| `argus/decision/bayesian.py` | 🔧 function | `sensitivity_analysis` | 337 |
| `argus/decision/calibration.py` | 🏛️ class | `CalibrationMetrics` | 25 |
| `argus/decision/calibration.py` | ⚙️ method | `CalibrationMetrics.summary` | 44 |
| `argus/decision/calibration.py` | 🔧 function | `compute_brier_score` | 54 |
| `argus/decision/calibration.py` | 🔧 function | `compute_ece` | 85 |
| `argus/decision/calibration.py` | 🔧 function | `compute_nll` | 151 |
| `argus/decision/calibration.py` | 🔧 function | `temperature_scaling` | 186 |
| `argus/decision/calibration.py` | ⚙️ method | `temperature_scaling.nll_objective` | 208 |
| `argus/decision/calibration.py` | 🔧 function | `apply_temperature` | 236 |
| `argus/decision/calibration.py` | 🏛️ class | `ReliabilityDiagram` | 262 |
| `argus/decision/calibration.py` | ⚙️ method | `ReliabilityDiagram.__init__` | 270 |
| `argus/decision/calibration.py` | ⚙️ method | `ReliabilityDiagram.add_sample` | 284 |
| `argus/decision/calibration.py` | ⚙️ method | `ReliabilityDiagram.add_batch` | 299 |
| `argus/decision/calibration.py` | ⚙️ method | `ReliabilityDiagram.compute` | 314 |
| `argus/decision/calibration.py` | ⚙️ method | `ReliabilityDiagram.get_diagram_data` | 339 |
| `argus/decision/calibration.py` | ⚙️ method | `ReliabilityDiagram.reset` | 364 |
| `argus/decision/calibration.py` | 🔧 function | `calibrate_model_outputs` | 370 |
| `argus/decision/eig.py` | 🏛️ class | `ActionCandidate` | 29 |
| `argus/decision/eig.py` | 🔧 function | `entropy` | 56 |
| `argus/decision/eig.py` | 🔧 function | `kl_divergence` | 73 |
| `argus/decision/eig.py` | 🏛️ class | `EIGEstimator` | 93 |
| `argus/decision/eig.py` | ⚙️ method | `EIGEstimator.__init__` | 110 |
| `argus/decision/eig.py` | ⚙️ method | `EIGEstimator.estimate` | 128 |
| `argus/decision/eig.py` | ⚙️ method | `EIGEstimator.estimate_discrete` | 173 |
| `argus/decision/eig.py` | ⚙️ method | `EIGEstimator.estimate_for_action` | 210 |
| `argus/decision/eig.py` | 🔧 function | `estimate_eig` | 269 |
| `argus/decision/eig.py` | 🔧 function | `rank_actions_by_eig` | 314 |
| `argus/decision/planner.py` | 🏛️ class | `PlannerConfig` | 31 |
| `argus/decision/planner.py` | 🏛️ class | `QueuedExperiment` | 52 |
| `argus/decision/planner.py` | ⚙️ method | `QueuedExperiment.__lt__` | 58 |
| `argus/decision/planner.py` | 🏛️ class | `ExperimentQueue` | 63 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.__init__` | 71 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.push` | 89 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.pop` | 134 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.peek` | 148 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.remove` | 159 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.get_all` | 179 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.__len__` | 184 |
| `argus/decision/planner.py` | ⚙️ method | `ExperimentQueue.remaining_budget` | 188 |
| `argus/decision/planner.py` | 🏛️ class | `VoIPlanner` | 195 |
| `argus/decision/planner.py` | ⚙️ method | `VoIPlanner.__init__` | 209 |
| `argus/decision/planner.py` | ⚙️ method | `VoIPlanner.plan` | 224 |
| `argus/decision/planner.py` | ⚙️ method | `VoIPlanner._apply_risk_aversion` | 289 |
| `argus/decision/planner.py` | ⚙️ method | `VoIPlanner._knapsack_selection` | 320 |
| `argus/decision/planner.py` | ⚙️ method | `VoIPlanner.what_if_analysis` | 376 |
| `argus/decision/planner.py` | ⚙️ method | `VoIPlanner.stopping_criteria_met` | 433 |
| `argus/durable/checkpointer.py` | 🏛️ class | `Checkpoint` | 23 |
| `argus/durable/checkpointer.py` | ⚙️ method | `Checkpoint.to_dict` | 33 |
| `argus/durable/checkpointer.py` | ⚙️ method | `Checkpoint.from_dict` | 39 |
| `argus/durable/checkpointer.py` | 🏛️ class | `BaseCheckpointer` | 45 |
| `argus/durable/checkpointer.py` | ⚙️ method | `BaseCheckpointer.save` | 49 |
| `argus/durable/checkpointer.py` | ⚙️ method | `BaseCheckpointer.load` | 53 |
| `argus/durable/checkpointer.py` | ⚙️ method | `BaseCheckpointer.list_checkpoints` | 57 |
| `argus/durable/checkpointer.py` | ⚙️ method | `BaseCheckpointer.delete` | 61 |
| `argus/durable/checkpointer.py` | 🏛️ class | `MemoryCheckpointer` | 65 |
| `argus/durable/checkpointer.py` | ⚙️ method | `MemoryCheckpointer.__init__` | 68 |
| `argus/durable/checkpointer.py` | ⚙️ method | `MemoryCheckpointer.save` | 73 |
| `argus/durable/checkpointer.py` | ⚙️ method | `MemoryCheckpointer.load` | 91 |
| `argus/durable/checkpointer.py` | ⚙️ method | `MemoryCheckpointer.list_checkpoints` | 99 |
| `argus/durable/checkpointer.py` | ⚙️ method | `MemoryCheckpointer.delete` | 103 |
| `argus/durable/checkpointer.py` | 🏛️ class | `SQLiteCheckpointer` | 112 |
| `argus/durable/checkpointer.py` | ⚙️ method | `SQLiteCheckpointer.__init__` | 115 |
| `argus/durable/checkpointer.py` | ⚙️ method | `SQLiteCheckpointer._init_db` | 120 |
| `argus/durable/checkpointer.py` | ⚙️ method | `SQLiteCheckpointer.save` | 135 |
| `argus/durable/checkpointer.py` | ⚙️ method | `SQLiteCheckpointer.load` | 151 |
| `argus/durable/checkpointer.py` | ⚙️ method | `SQLiteCheckpointer.list_checkpoints` | 168 |
| `argus/durable/checkpointer.py` | ⚙️ method | `SQLiteCheckpointer.delete` | 179 |
| `argus/durable/checkpointer.py` | 🏛️ class | `FileSystemCheckpointer` | 185 |
| `argus/durable/checkpointer.py` | ⚙️ method | `FileSystemCheckpointer.__init__` | 188 |
| `argus/durable/checkpointer.py` | ⚙️ method | `FileSystemCheckpointer._get_thread_dir` | 192 |
| `argus/durable/checkpointer.py` | ⚙️ method | `FileSystemCheckpointer.save` | 195 |
| `argus/durable/checkpointer.py` | ⚙️ method | `FileSystemCheckpointer.load` | 208 |
| `argus/durable/checkpointer.py` | ⚙️ method | `FileSystemCheckpointer.list_checkpoints` | 223 |
| `argus/durable/checkpointer.py` | ⚙️ method | `FileSystemCheckpointer.delete` | 230 |
| `argus/durable/config.py` | 🏛️ class | `CheckpointerType` | 18 |
| `argus/durable/config.py` | 🏛️ class | `RetryPolicy` | 25 |
| `argus/durable/config.py` | 🏛️ class | `DurableConfig` | 33 |
| `argus/durable/config.py` | 🔧 function | `get_default_durable_config` | 46 |
| `argus/durable/state.py` | 🏛️ class | `StateSnapshot` | 22 |
| `argus/durable/state.py` | ⚙️ method | `StateSnapshot.to_dict` | 30 |
| `argus/durable/state.py` | ⚙️ method | `StateSnapshot.from_dict` | 36 |
| `argus/durable/state.py` | 🏛️ class | `DebateState` | 43 |
| `argus/durable/state.py` | ⚙️ method | `DebateState.to_dict` | 58 |
| `argus/durable/state.py` | ⚙️ method | `DebateState.from_dict` | 65 |
| `argus/durable/state.py` | 🔧 function | `serialize_state` | 72 |
| `argus/durable/state.py` | 🔧 function | `deserialize_state` | 77 |
| `argus/durable/state.py` | 🔧 function | `serialize_graph` | 82 |
| `argus/durable/state.py` | 🏛️ class | `StateManager` | 111 |
| `argus/durable/state.py` | ⚙️ method | `StateManager.__init__` | 114 |
| `argus/durable/state.py` | ⚙️ method | `StateManager.initialize` | 119 |
| `argus/durable/state.py` | ⚙️ method | `StateManager.get_state` | 126 |
| `argus/durable/state.py` | ⚙️ method | `StateManager.update` | 129 |
| `argus/durable/state.py` | ⚙️ method | `StateManager.snapshot` | 139 |
| `argus/durable/state.py` | ⚙️ method | `StateManager.restore` | 152 |
| `argus/durable/state.py` | ⚙️ method | `StateManager.get_snapshots` | 157 |
| `argus/durable/tasks.py` | 🏛️ class | `TaskResult` | 21 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskResult.to_dict` | 30 |
| `argus/durable/tasks.py` | 🏛️ class | `TaskRegistry` | 41 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.__init__` | 44 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry._compute_task_id` | 48 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.is_executed` | 53 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.get_result` | 56 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.record` | 59 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.mark_pending` | 63 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.is_pending` | 66 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.clear` | 69 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskRegistry.get_all_results` | 73 |
| `argus/durable/tasks.py` | 🔧 function | `get_task_registry` | 81 |
| `argus/durable/tasks.py` | 🔧 function | `idempotent_task` | 85 |
| `argus/durable/tasks.py` | ⚙️ method | `idempotent_task.decorator` | 100 |
| `argus/durable/tasks.py` | ⚙️ method | `idempotent_task.decorator.wrapper` | 102 |
| `argus/durable/tasks.py` | 🏛️ class | `TaskExecutor` | 141 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskExecutor.__init__` | 144 |
| `argus/durable/tasks.py` | ⚙️ method | `TaskExecutor.execute` | 148 |
| `argus/durable/workflow.py` | 🏛️ class | `WorkflowStatus` | 27 |
| `argus/durable/workflow.py` | 🏛️ class | `WorkflowRun` | 38 |
| `argus/durable/workflow.py` | 🏛️ class | `DurableWorkflow` | 51 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.__init__` | 70 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.step` | 84 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.checkpoint` | 105 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.resume` | 121 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.rollback` | 150 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.start_run` | 159 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.complete_run` | 171 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.get_current_run` | 181 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableWorkflow.list_checkpoints` | 184 |
| `argus/durable/workflow.py` | 🏛️ class | `DurableDebateWorkflow` | 188 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableDebateWorkflow.__init__` | 191 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableDebateWorkflow.initialize_debate` | 195 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableDebateWorkflow.run_round` | 206 |
| `argus/durable/workflow.py` | ⚙️ method | `DurableDebateWorkflow.run_round.execute_round` | 208 |
| `argus/evaluation/benchmarks/base.py` | 🏛️ class | `BenchmarkStatus` | 24 |
| `argus/evaluation/benchmarks/base.py` | 🏛️ class | `BenchmarkConfig` | 34 |
| `argus/evaluation/benchmarks/base.py` | ⚙️ method | `BenchmarkConfig.__post_init__` | 54 |
| `argus/evaluation/benchmarks/base.py` | 🏛️ class | `SampleResult` | 60 |
| `argus/evaluation/benchmarks/base.py` | 🏛️ class | `BenchmarkResult` | 86 |
| `argus/evaluation/benchmarks/base.py` | ⚙️ method | `BenchmarkResult.duration_seconds` | 121 |
| `argus/evaluation/benchmarks/base.py` | ⚙️ method | `BenchmarkResult.to_dict` | 127 |
| `argus/evaluation/benchmarks/base.py` | 🏛️ class | `Benchmark` | 146 |
| `argus/evaluation/benchmarks/base.py` | ⚙️ method | `Benchmark.__init__` | 162 |
| `argus/evaluation/benchmarks/base.py` | ⚙️ method | `Benchmark.evaluate_sample` | 173 |
| `argus/evaluation/benchmarks/base.py` | ⚙️ method | `Benchmark.run` | 191 |
| `argus/evaluation/benchmarks/base.py` | ⚙️ method | `Benchmark._aggregate_scores` | 269 |
| `argus/evaluation/benchmarks/debate_quality.py` | 🏛️ class | `DebateQualityBenchmark` | 23 |
| `argus/evaluation/benchmarks/debate_quality.py` | ⚙️ method | `DebateQualityBenchmark.__init__` | 38 |
| `argus/evaluation/benchmarks/debate_quality.py` | ⚙️ method | `DebateQualityBenchmark.evaluate_sample` | 44 |
| `argus/evaluation/benchmarks/debate_quality.py` | ⚙️ method | `DebateQualityBenchmark._check_verdict_match` | 125 |
| `argus/evaluation/benchmarks/debate_quality.py` | ⚙️ method | `DebateQualityBenchmark._check_verdict_match.normalize` | 142 |
| `argus/evaluation/benchmarks/evidence_analysis.py` | 🏛️ class | `EvidenceAnalysisBenchmark` | 22 |
| `argus/evaluation/benchmarks/evidence_analysis.py` | ⚙️ method | `EvidenceAnalysisBenchmark.__init__` | 36 |
| `argus/evaluation/benchmarks/evidence_analysis.py` | ⚙️ method | `EvidenceAnalysisBenchmark.evaluate_sample` | 42 |
| `argus/evaluation/benchmarks/evidence_analysis.py` | ⚙️ method | `EvidenceAnalysisBenchmark._analyze_evidence` | 125 |
| `argus/evaluation/benchmarks/reasoning_depth.py` | 🏛️ class | `ReasoningDepthBenchmark` | 23 |
| `argus/evaluation/benchmarks/reasoning_depth.py` | ⚙️ method | `ReasoningDepthBenchmark.__init__` | 37 |
| `argus/evaluation/benchmarks/reasoning_depth.py` | ⚙️ method | `ReasoningDepthBenchmark.evaluate_sample` | 43 |
| `argus/evaluation/benchmarks/reasoning_depth.py` | ⚙️ method | `ReasoningDepthBenchmark._analyze_reasoning` | 126 |
| `argus/evaluation/datasets/global_benchmarks.py` | 🏛️ class | `HFBenchmarkInfo` | 41 |
| `argus/evaluation/datasets/global_benchmarks.py` | 🏛️ class | `HuggingFaceDatasetLoader` | 205 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader.__init__` | 217 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader._check_datasets_available` | 226 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader.list_benchmarks` | 238 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader.get_benchmark_info` | 242 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader.load` | 253 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader._convert_to_argus_format` | 335 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader._get_field` | 417 |
| `argus/evaluation/datasets/global_benchmarks.py` | ⚙️ method | `HuggingFaceDatasetLoader.to_csv` | 440 |
| `argus/evaluation/datasets/global_benchmarks.py` | 🔧 function | `list_global_benchmarks` | 468 |
| `argus/evaluation/datasets/global_benchmarks.py` | 🔧 function | `load_global_benchmark` | 477 |
| `argus/evaluation/datasets/global_benchmarks.py` | 🔧 function | `get_benchmark_info` | 533 |
| `argus/evaluation/datasets/global_benchmarks.py` | 🔧 function | `download_all_benchmarks` | 545 |
| `argus/evaluation/datasets/loader.py` | 🏛️ class | `DatasetInfo` | 22 |
| `argus/evaluation/datasets/loader.py` | ⚙️ method | `DatasetInfo.__post_init__` | 40 |
| `argus/evaluation/datasets/loader.py` | 🔧 function | `list_datasets` | 110 |
| `argus/evaluation/datasets/loader.py` | 🔧 function | `get_dataset_path` | 119 |
| `argus/evaluation/datasets/loader.py` | 🔧 function | `load_dataset` | 138 |
| `argus/evaluation/datasets/loader.py` | 🔧 function | `validate_dataset` | 172 |
| `argus/evaluation/datasets/loader.py` | 🔧 function | `validate_all` | 231 |
| `argus/evaluation/runner/benchmark_runner.py` | 🏛️ class | `RunConfig` | 32 |
| `argus/evaluation/runner/benchmark_runner.py` | ⚙️ method | `RunConfig.__post_init__` | 54 |
| `argus/evaluation/runner/benchmark_runner.py` | 🏛️ class | `BenchmarkRunner` | 58 |
| `argus/evaluation/runner/benchmark_runner.py` | ⚙️ method | `BenchmarkRunner.__init__` | 71 |
| `argus/evaluation/runner/benchmark_runner.py` | ⚙️ method | `BenchmarkRunner.run` | 83 |
| `argus/evaluation/runner/benchmark_runner.py` | ⚙️ method | `BenchmarkRunner._get_benchmarks` | 145 |
| `argus/evaluation/runner/benchmark_runner.py` | ⚙️ method | `BenchmarkRunner._dry_run` | 175 |
| `argus/evaluation/runner/benchmark_runner.py` | ⚙️ method | `BenchmarkRunner._save_result` | 217 |
| `argus/evaluation/runner/benchmark_runner.py` | ⚙️ method | `BenchmarkRunner._save_summary` | 231 |
| `argus/evaluation/runner/benchmark_runner.py` | 🔧 function | `main` | 246 |
| `argus/evaluation/runner/report_generator.py` | 🏛️ class | `ReportGenerator` | 21 |
| `argus/evaluation/runner/report_generator.py` | ⚙️ method | `ReportGenerator.__init__` | 32 |
| `argus/evaluation/runner/report_generator.py` | ⚙️ method | `ReportGenerator.generate_markdown` | 41 |
| `argus/evaluation/runner/report_generator.py` | ⚙️ method | `ReportGenerator.generate_json` | 114 |
| `argus/evaluation/runner/report_generator.py` | ⚙️ method | `ReportGenerator.generate_html` | 139 |
| `argus/evaluation/runner/report_generator.py` | 🔧 function | `generate_report` | 243 |
| `argus/evaluation/runner/results_aggregator.py` | 🏛️ class | `AggregatedMetrics` | 23 |
| `argus/evaluation/runner/results_aggregator.py` | ⚙️ method | `AggregatedMetrics.to_dict` | 41 |
| `argus/evaluation/runner/results_aggregator.py` | 🏛️ class | `ResultsAggregator` | 52 |
| `argus/evaluation/runner/results_aggregator.py` | ⚙️ method | `ResultsAggregator.__init__` | 61 |
| `argus/evaluation/runner/results_aggregator.py` | ⚙️ method | `ResultsAggregator.add_result` | 65 |
| `argus/evaluation/runner/results_aggregator.py` | ⚙️ method | `ResultsAggregator.add_results` | 73 |
| `argus/evaluation/runner/results_aggregator.py` | ⚙️ method | `ResultsAggregator.load_from_directory` | 81 |
| `argus/evaluation/runner/results_aggregator.py` | ⚙️ method | `ResultsAggregator.summarize` | 105 |
| `argus/evaluation/runner/results_aggregator.py` | ⚙️ method | `ResultsAggregator.compare_runs` | 164 |
| `argus/evaluation/runner/results_aggregator.py` | 🔧 function | `aggregate_results` | 197 |
| `argus/evaluation/scoring/aggregate.py` | 🏛️ class | `CompositeScore` | 22 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `CompositeScore.__post_init__` | 36 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `CompositeScore._compute` | 45 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `CompositeScore.with_weights` | 60 |
| `argus/evaluation/scoring/aggregate.py` | 🏛️ class | `ScoreComparison` | 66 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `ScoreComparison.__post_init__` | 82 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `ScoreComparison._compute_differences` | 85 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `ScoreComparison.composite_change` | 110 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `ScoreComparison.to_dict` | 114 |
| `argus/evaluation/scoring/aggregate.py` | ⚙️ method | `ScoreComparison.__str__` | 125 |
| `argus/evaluation/scoring/aggregate.py` | 🔧 function | `generate_score_report` | 144 |
| `argus/evaluation/scoring/aggregate.py` | 🔧 function | `_generate_markdown_report` | 179 |
| `argus/evaluation/scoring/aggregate.py` | 🔧 function | `_generate_html_report` | 220 |
| `argus/evaluation/scoring/metrics.py` | 🏛️ class | `MetricDefinition` | 36 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_arcis` | 135 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_evid_q` | 241 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_dialec` | 309 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_rebut_f` | 380 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_conv_s` | 451 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_prov_i` | 534 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_calib_m` | 609 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_eig_u` | 674 |
| `argus/evaluation/scoring/metrics.py` | 🔧 function | `compute_all_scores` | 746 |
| `argus/evaluation/scoring/metrics.py` | 🏛️ class | `ScoreCard` | 769 |
| `argus/evaluation/scoring/metrics.py` | ⚙️ method | `ScoreCard.__post_init__` | 783 |
| `argus/evaluation/scoring/metrics.py` | ⚙️ method | `ScoreCard._compute_composite` | 787 |
| `argus/evaluation/scoring/metrics.py` | ⚙️ method | `ScoreCard.to_dict` | 800 |
| `argus/evaluation/scoring/metrics.py` | ⚙️ method | `ScoreCard.__str__` | 809 |
| `argus/evaluation/scoring/metrics.py` | ⚙️ method | `ScoreCard.from_result` | 819 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_accuracy` | 27 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_precision_recall_f1` | 51 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_macro_f1` | 91 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_brier_score` | 122 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_ece` | 148 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_mce` | 200 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_cross_entropy` | 231 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_log_loss` | 258 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_argument_coverage` | 279 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_dialectical_balance` | 300 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_verdict_confidence_correlation` | 327 |
| `argus/evaluation/scoring/standard_metrics.py` | 🏛️ class | `StandardMetricsResult` | 365 |
| `argus/evaluation/scoring/standard_metrics.py` | ⚙️ method | `StandardMetricsResult.to_dict` | 379 |
| `argus/evaluation/scoring/standard_metrics.py` | 🔧 function | `compute_all_standard_metrics` | 395 |
| `argus/hitl/callbacks.py` | 🏛️ class | `FeedbackType` | 19 |
| `argus/hitl/callbacks.py` | 🏛️ class | `Feedback` | 28 |
| `argus/hitl/callbacks.py` | ⚙️ method | `Feedback.to_dict` | 38 |
| `argus/hitl/callbacks.py` | 🏛️ class | `BaseCallback` | 50 |
| `argus/hitl/callbacks.py` | ⚙️ method | `BaseCallback.__init__` | 53 |
| `argus/hitl/callbacks.py` | ⚙️ method | `BaseCallback.collect` | 57 |
| `argus/hitl/callbacks.py` | ⚙️ method | `BaseCallback.get_feedback` | 60 |
| `argus/hitl/callbacks.py` | 🏛️ class | `FeedbackCallback` | 64 |
| `argus/hitl/callbacks.py` | ⚙️ method | `FeedbackCallback.__init__` | 67 |
| `argus/hitl/callbacks.py` | ⚙️ method | `FeedbackCallback.collect` | 71 |
| `argus/hitl/callbacks.py` | 🏛️ class | `RatingCallback` | 95 |
| `argus/hitl/callbacks.py` | ⚙️ method | `RatingCallback.__init__` | 98 |
| `argus/hitl/callbacks.py` | ⚙️ method | `RatingCallback.collect` | 103 |
| `argus/hitl/callbacks.py` | ⚙️ method | `RatingCallback.get_average_rating` | 126 |
| `argus/hitl/callbacks.py` | 🏛️ class | `AnnotationCallback` | 133 |
| `argus/hitl/callbacks.py` | ⚙️ method | `AnnotationCallback.collect` | 136 |
| `argus/hitl/callbacks.py` | 🏛️ class | `CorrectionCallback` | 158 |
| `argus/hitl/callbacks.py` | ⚙️ method | `CorrectionCallback.collect` | 161 |
| `argus/hitl/callbacks.py` | ⚙️ method | `CorrectionCallback.get_corrections_for_agent` | 183 |
| `argus/hitl/callbacks.py` | 🏛️ class | `CallbackManager` | 187 |
| `argus/hitl/callbacks.py` | ⚙️ method | `CallbackManager.__init__` | 190 |
| `argus/hitl/callbacks.py` | ⚙️ method | `CallbackManager.get_all_feedback` | 196 |
| `argus/hitl/config.py` | 🏛️ class | `ApprovalMode` | 19 |
| `argus/hitl/config.py` | 🏛️ class | `InterruptionPoint` | 29 |
| `argus/hitl/config.py` | 🏛️ class | `SensitivityLevel` | 45 |
| `argus/hitl/config.py` | 🏛️ class | `FeedbackType` | 56 |
| `argus/hitl/config.py` | 🏛️ class | `HITLCallbackConfig` | 65 |
| `argus/hitl/config.py` | 🏛️ class | `HITLConfig` | 89 |
| `argus/hitl/config.py` | ⚙️ method | `HITLConfig.should_require_approval` | 163 |
| `argus/hitl/config.py` | ⚙️ method | `HITLConfig.should_interrupt_at` | 194 |
| `argus/hitl/config.py` | 🔧 function | `get_default_hitl_config` | 209 |
| `argus/hitl/handlers.py` | 🏛️ class | `HandlerResult` | 22 |
| `argus/hitl/handlers.py` | ⚙️ method | `HandlerResult.approved` | 32 |
| `argus/hitl/handlers.py` | ⚙️ method | `HandlerResult.rejected` | 36 |
| `argus/hitl/handlers.py` | ⚙️ method | `HandlerResult.modified` | 40 |
| `argus/hitl/handlers.py` | 🏛️ class | `BaseHandler` | 44 |
| `argus/hitl/handlers.py` | ⚙️ method | `BaseHandler.__init__` | 47 |
| `argus/hitl/handlers.py` | ⚙️ method | `BaseHandler.handle` | 53 |
| `argus/hitl/handlers.py` | ⚙️ method | `BaseHandler.record` | 56 |
| `argus/hitl/handlers.py` | 🏛️ class | `ApprovalHandler` | 66 |
| `argus/hitl/handlers.py` | ⚙️ method | `ApprovalHandler.__init__` | 69 |
| `argus/hitl/handlers.py` | ⚙️ method | `ApprovalHandler.handle` | 73 |
| `argus/hitl/handlers.py` | 🏛️ class | `RejectionHandler` | 84 |
| `argus/hitl/handlers.py` | ⚙️ method | `RejectionHandler.__init__` | 87 |
| `argus/hitl/handlers.py` | ⚙️ method | `RejectionHandler.handle` | 92 |
| `argus/hitl/handlers.py` | 🏛️ class | `ModificationHandler` | 104 |
| `argus/hitl/handlers.py` | ⚙️ method | `ModificationHandler.__init__` | 107 |
| `argus/hitl/handlers.py` | ⚙️ method | `ModificationHandler.handle` | 111 |
| `argus/hitl/handlers.py` | 🏛️ class | `EscalationHandler` | 121 |
| `argus/hitl/handlers.py` | ⚙️ method | `EscalationHandler.__init__` | 124 |
| `argus/hitl/handlers.py` | ⚙️ method | `EscalationHandler.handle` | 129 |
| `argus/hitl/handlers.py` | 🏛️ class | `DecisionRouter` | 143 |
| `argus/hitl/handlers.py` | ⚙️ method | `DecisionRouter.__init__` | 146 |
| `argus/hitl/handlers.py` | ⚙️ method | `DecisionRouter.route` | 153 |
| `argus/hitl/middleware.py` | 🏛️ class | `InterruptStatus` | 36 |
| `argus/hitl/middleware.py` | 🏛️ class | `InterruptRequest` | 48 |
| `argus/hitl/middleware.py` | ⚙️ method | `InterruptRequest.to_dict` | 80 |
| `argus/hitl/middleware.py` | ⚙️ method | `InterruptRequest.from_dict` | 91 |
| `argus/hitl/middleware.py` | 🏛️ class | `MiddlewareState` | 104 |
| `argus/hitl/middleware.py` | ⚙️ method | `MiddlewareState.to_json` | 116 |
| `argus/hitl/middleware.py` | ⚙️ method | `MiddlewareState.from_json` | 129 |
| `argus/hitl/middleware.py` | 🏛️ class | `HITLMiddleware` | 137 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.__init__` | 156 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.should_intercept` | 177 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.create_interrupt` | 201 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.save_state` | 245 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.restore_state` | 280 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.submit_response` | 301 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.get_pending_requests` | 347 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.get_request` | 356 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.check_timeout` | 368 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.get_decision_history` | 393 |
| `argus/hitl/middleware.py` | ⚙️ method | `HITLMiddleware.clear_pending` | 415 |
| `argus/hitl/middleware.py` | 🏛️ class | `ToolInterceptor` | 429 |
| `argus/hitl/middleware.py` | ⚙️ method | `ToolInterceptor.__init__` | 443 |
| `argus/hitl/middleware.py` | ⚙️ method | `ToolInterceptor.wrap` | 451 |
| `argus/hitl/middleware.py` | ⚙️ method | `ToolInterceptor.wrap.decorator` | 465 |
| `argus/hitl/middleware.py` | ⚙️ method | `ToolInterceptor.wrap.decorator.wrapper` | 466 |
| `argus/knowledge/chunking.py` | 🏛️ class | `ChunkingStrategy` | 21 |
| `argus/knowledge/chunking.py` | 🏛️ class | `ChunkingConfig` | 31 |
| `argus/knowledge/chunking.py` | 🏛️ class | `Chunker` | 49 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker.__init__` | 63 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker.chunk` | 90 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker._estimate_tokens` | 111 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker._create_chunk` | 115 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker._chunk_fixed` | 133 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker._chunk_sentences` | 171 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker._chunk_paragraphs` | 216 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker._chunk_recursive` | 262 |
| `argus/knowledge/chunking.py` | ⚙️ method | `Chunker._chunk_recursive.split_text` | 283 |
| `argus/knowledge/chunking.py` | 🔧 function | `chunk_document` | 361 |
| `argus/knowledge/connectors/arxiv.py` | 🏛️ class | `ArxivPaper` | 35 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivPaper.to_document` | 50 |
| `argus/knowledge/connectors/arxiv.py` | 🏛️ class | `ArxivConnectorConfig` | 81 |
| `argus/knowledge/connectors/arxiv.py` | 🏛️ class | `ArxivConnector` | 103 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector.__init__` | 144 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector.arxiv_config` | 154 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector._get_client` | 158 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector._build_query` | 174 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector._parse_paper` | 229 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector.fetch` | 256 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector.fetch_by_id` | 360 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector.fetch_by_category` | 373 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector.test_connection` | 398 |
| `argus/knowledge/connectors/arxiv.py` | ⚙️ method | `ArxivConnector.validate_config` | 410 |
| `argus/knowledge/connectors/base.py` | 🏛️ class | `ConnectorConfig` | 37 |
| `argus/knowledge/connectors/base.py` | 🏛️ class | `ConnectorResult` | 70 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorResult.to_dict` | 86 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorResult.from_error` | 97 |
| `argus/knowledge/connectors/base.py` | 🏛️ class | `BaseConnector` | 102 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.__init__` | 126 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.fetch` | 139 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.test_connection` | 159 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.validate_config` | 169 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector._rate_limit_wait` | 179 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.__call__` | 192 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.get_schema` | 235 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.get_stats` | 248 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `BaseConnector.__repr__` | 260 |
| `argus/knowledge/connectors/base.py` | 🏛️ class | `ConnectorRegistry` | 264 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.__init__` | 275 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.register` | 281 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.unregister` | 299 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.get` | 314 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.list_all` | 326 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.get_all` | 335 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.fetch_from_all` | 344 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.__len__` | 366 |
| `argus/knowledge/connectors/base.py` | ⚙️ method | `ConnectorRegistry.__contains__` | 369 |
| `argus/knowledge/connectors/base.py` | 🔧 function | `get_default_registry` | 381 |
| `argus/knowledge/connectors/base.py` | 🔧 function | `register_connector` | 395 |
| `argus/knowledge/connectors/crossref.py` | 🏛️ class | `CrossRefWork` | 36 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefWork.author_names` | 58 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefWork.citation_string` | 73 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefWork.to_document` | 111 |
| `argus/knowledge/connectors/crossref.py` | 🏛️ class | `CrossRefConnectorConfig` | 153 |
| `argus/knowledge/connectors/crossref.py` | 🏛️ class | `CrossRefConnector` | 175 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.__init__` | 200 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.crossref_config` | 210 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector._get_session` | 214 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector._make_request` | 233 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector._parse_date` | 275 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector._parse_work` | 297 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.fetch` | 350 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.fetch_by_doi` | 422 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.fetch_references` | 463 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.fetch_citing_works` | 536 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.test_connection` | 558 |
| `argus/knowledge/connectors/crossref.py` | ⚙️ method | `CrossRefConnector.close` | 570 |
| `argus/knowledge/connectors/web.py` | 🏛️ class | `RobotRule` | 40 |
| `argus/knowledge/connectors/web.py` | 🏛️ class | `RobotsTxtRules` | 47 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `RobotsTxtRules.is_allowed` | 54 |
| `argus/knowledge/connectors/web.py` | 🏛️ class | `RobotsTxtParser` | 106 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `RobotsTxtParser.__init__` | 120 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `RobotsTxtParser.parse` | 129 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `RobotsTxtParser.fetch_and_parse` | 212 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `RobotsTxtParser.is_allowed` | 266 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `RobotsTxtParser.clear_cache` | 293 |
| `argus/knowledge/connectors/web.py` | 🏛️ class | `WebConnectorConfig` | 302 |
| `argus/knowledge/connectors/web.py` | 🏛️ class | `WebConnector` | 345 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector.__init__` | 366 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector.web_config` | 379 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector._get_session` | 383 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector._check_robots_txt` | 397 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector._get_crawl_delay` | 416 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector.fetch` | 436 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector.fetch_multiple` | 548 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector._parse_html` | 570 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector.get_sitemaps` | 646 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector.test_connection` | 662 |
| `argus/knowledge/connectors/web.py` | ⚙️ method | `WebConnector.close` | 677 |
| `argus/knowledge/embeddings.py` | 🏛️ class | `EmbeddingGenerator` | 23 |
| `argus/knowledge/embeddings.py` | ⚙️ method | `EmbeddingGenerator.__init__` | 36 |
| `argus/knowledge/embeddings.py` | ⚙️ method | `EmbeddingGenerator._load_model` | 60 |
| `argus/knowledge/embeddings.py` | ⚙️ method | `EmbeddingGenerator.dimension` | 84 |
| `argus/knowledge/embeddings.py` | ⚙️ method | `EmbeddingGenerator.embed_texts` | 89 |
| `argus/knowledge/embeddings.py` | ⚙️ method | `EmbeddingGenerator.embed_query` | 111 |
| `argus/knowledge/embeddings.py` | ⚙️ method | `EmbeddingGenerator.embed_chunks` | 131 |
| `argus/knowledge/embeddings.py` | 🔧 function | `generate_embeddings` | 170 |
| `argus/knowledge/embeddings.py` | 🔧 function | `cosine_similarity` | 188 |
| `argus/knowledge/embeddings.py` | 🔧 function | `batch_cosine_similarity` | 212 |
| `argus/knowledge/indexing.py` | 🏛️ class | `SearchResult` | 22 |
| `argus/knowledge/indexing.py` | 🏛️ class | `BM25Index` | 31 |
| `argus/knowledge/indexing.py` | ⚙️ method | `BM25Index.__init__` | 43 |
| `argus/knowledge/indexing.py` | ⚙️ method | `BM25Index._tokenize` | 63 |
| `argus/knowledge/indexing.py` | ⚙️ method | `BM25Index.add_chunks` | 71 |
| `argus/knowledge/indexing.py` | ⚙️ method | `BM25Index.search` | 95 |
| `argus/knowledge/indexing.py` | ⚙️ method | `BM25Index.clear` | 132 |
| `argus/knowledge/indexing.py` | 🏛️ class | `VectorIndex` | 140 |
| `argus/knowledge/indexing.py` | ⚙️ method | `VectorIndex.__init__` | 152 |
| `argus/knowledge/indexing.py` | ⚙️ method | `VectorIndex._init_index` | 173 |
| `argus/knowledge/indexing.py` | ⚙️ method | `VectorIndex.add_vectors` | 193 |
| `argus/knowledge/indexing.py` | ⚙️ method | `VectorIndex.search` | 232 |
| `argus/knowledge/indexing.py` | ⚙️ method | `VectorIndex.clear` | 270 |
| `argus/knowledge/indexing.py` | 🏛️ class | `HybridIndex` | 277 |
| `argus/knowledge/indexing.py` | ⚙️ method | `HybridIndex.__init__` | 290 |
| `argus/knowledge/indexing.py` | ⚙️ method | `HybridIndex.add_chunks` | 310 |
| `argus/knowledge/indexing.py` | ⚙️ method | `HybridIndex.search` | 337 |
| `argus/knowledge/indexing.py` | ⚙️ method | `HybridIndex.search.normalize` | 365 |
| `argus/knowledge/indexing.py` | ⚙️ method | `HybridIndex.clear` | 407 |
| `argus/knowledge/ingestion.py` | 🔧 function | `compute_checksum` | 26 |
| `argus/knowledge/ingestion.py` | 🔧 function | `detect_source_type` | 31 |
| `argus/knowledge/ingestion.py` | 🏛️ class | `DocumentLoader` | 59 |
| `argus/knowledge/ingestion.py` | ⚙️ method | `DocumentLoader.__init__` | 73 |
| `argus/knowledge/ingestion.py` | ⚙️ method | `DocumentLoader.load` | 88 |
| `argus/knowledge/ingestion.py` | ⚙️ method | `DocumentLoader._load_pdf` | 119 |
| `argus/knowledge/ingestion.py` | ⚙️ method | `DocumentLoader._load_html` | 174 |
| `argus/knowledge/ingestion.py` | ⚙️ method | `DocumentLoader._load_csv` | 241 |
| `argus/knowledge/ingestion.py` | ⚙️ method | `DocumentLoader._load_json` | 278 |
| `argus/knowledge/ingestion.py` | ⚙️ method | `DocumentLoader._load_text` | 306 |
| `argus/knowledge/ingestion.py` | 🔧 function | `load_document` | 345 |
| `argus/knowledge/ingestion.py` | 🔧 function | `load_pdf` | 365 |
| `argus/knowledge/ingestion.py` | 🔧 function | `load_html` | 374 |
| `argus/knowledge/ingestion.py` | 🔧 function | `load_text` | 383 |
| `argus/mcp/client.py` | 🏛️ class | `MCPToolInfo` | 22 |
| `argus/mcp/client.py` | 🏛️ class | `MCPResourceInfo` | 30 |
| `argus/mcp/client.py` | 🏛️ class | `MCPClient` | 38 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.__init__` | 48 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.connect` | 57 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient._send_request` | 75 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.list_tools` | 92 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.call_tool` | 105 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.list_resources` | 119 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.read_resource` | 133 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.disconnect` | 146 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.is_connected` | 156 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.__enter__` | 159 |
| `argus/mcp/client.py` | ⚙️ method | `MCPClient.__exit__` | 162 |
| `argus/mcp/config.py` | 🏛️ class | `TransportType` | 18 |
| `argus/mcp/config.py` | 🏛️ class | `MCPServerConfig` | 25 |
| `argus/mcp/config.py` | 🏛️ class | `MCPClientConfig` | 39 |
| `argus/mcp/config.py` | 🏛️ class | `MCPToolSchema` | 48 |
| `argus/mcp/config.py` | 🏛️ class | `MCPResourceSchema` | 56 |
| `argus/mcp/config.py` | 🔧 function | `get_default_server_config` | 64 |
| `argus/mcp/config.py` | 🔧 function | `get_default_client_config` | 68 |
| `argus/mcp/resources.py` | 🏛️ class | `BaseResourceAdapter` | 23 |
| `argus/mcp/resources.py` | ⚙️ method | `BaseResourceAdapter.uri` | 28 |
| `argus/mcp/resources.py` | ⚙️ method | `BaseResourceAdapter.name` | 33 |
| `argus/mcp/resources.py` | ⚙️ method | `BaseResourceAdapter.description` | 37 |
| `argus/mcp/resources.py` | ⚙️ method | `BaseResourceAdapter.mime_type` | 41 |
| `argus/mcp/resources.py` | ⚙️ method | `BaseResourceAdapter.read` | 45 |
| `argus/mcp/resources.py` | ⚙️ method | `BaseResourceAdapter.get_schema` | 48 |
| `argus/mcp/resources.py` | 🏛️ class | `CDAGResource` | 55 |
| `argus/mcp/resources.py` | ⚙️ method | `CDAGResource.__init__` | 58 |
| `argus/mcp/resources.py` | ⚙️ method | `CDAGResource.uri` | 63 |
| `argus/mcp/resources.py` | ⚙️ method | `CDAGResource.name` | 67 |
| `argus/mcp/resources.py` | ⚙️ method | `CDAGResource.description` | 71 |
| `argus/mcp/resources.py` | ⚙️ method | `CDAGResource.read` | 74 |
| `argus/mcp/resources.py` | 🏛️ class | `EvidenceResource` | 95 |
| `argus/mcp/resources.py` | ⚙️ method | `EvidenceResource.__init__` | 98 |
| `argus/mcp/resources.py` | ⚙️ method | `EvidenceResource.uri` | 103 |
| `argus/mcp/resources.py` | ⚙️ method | `EvidenceResource.name` | 107 |
| `argus/mcp/resources.py` | ⚙️ method | `EvidenceResource.description` | 111 |
| `argus/mcp/resources.py` | ⚙️ method | `EvidenceResource.read` | 114 |
| `argus/mcp/resources.py` | 🏛️ class | `ProvenanceResource` | 127 |
| `argus/mcp/resources.py` | ⚙️ method | `ProvenanceResource.__init__` | 130 |
| `argus/mcp/resources.py` | ⚙️ method | `ProvenanceResource.uri` | 134 |
| `argus/mcp/resources.py` | ⚙️ method | `ProvenanceResource.name` | 138 |
| `argus/mcp/resources.py` | ⚙️ method | `ProvenanceResource.description` | 142 |
| `argus/mcp/resources.py` | ⚙️ method | `ProvenanceResource.read` | 145 |
| `argus/mcp/resources.py` | 🏛️ class | `ConfigResource` | 153 |
| `argus/mcp/resources.py` | ⚙️ method | `ConfigResource.__init__` | 156 |
| `argus/mcp/resources.py` | ⚙️ method | `ConfigResource.uri` | 160 |
| `argus/mcp/resources.py` | ⚙️ method | `ConfigResource.name` | 164 |
| `argus/mcp/resources.py` | ⚙️ method | `ConfigResource.description` | 168 |
| `argus/mcp/resources.py` | ⚙️ method | `ConfigResource.read` | 171 |
| `argus/mcp/resources.py` | 🏛️ class | `ResourceRegistry` | 182 |
| `argus/mcp/resources.py` | ⚙️ method | `ResourceRegistry.__init__` | 185 |
| `argus/mcp/resources.py` | ⚙️ method | `ResourceRegistry.register` | 188 |
| `argus/mcp/resources.py` | ⚙️ method | `ResourceRegistry.get` | 192 |
| `argus/mcp/resources.py` | ⚙️ method | `ResourceRegistry.list_all` | 195 |
| `argus/mcp/resources.py` | ⚙️ method | `ResourceRegistry.read` | 198 |
| `argus/mcp/server.py` | 🏛️ class | `MCPRequest` | 25 |
| `argus/mcp/server.py` | 🏛️ class | `MCPResponse` | 34 |
| `argus/mcp/server.py` | ⚙️ method | `MCPResponse.to_dict` | 41 |
| `argus/mcp/server.py` | 🏛️ class | `ArgusServer` | 50 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.__init__` | 69 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.tool` | 78 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.tool.decorator` | 80 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.resource` | 90 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.resource.decorator` | 92 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.register_argus_tool` | 103 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.register_argus_tool.wrapper` | 105 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer._generate_tool_schema` | 115 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.handle_request` | 138 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer._handle_initialize` | 157 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer._handle_tools_list` | 167 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer._handle_tools_call` | 172 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer._handle_resources_list` | 183 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer._handle_resources_read` | 188 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.run` | 198 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer._run_stdio` | 205 |
| `argus/mcp/server.py` | ⚙️ method | `ArgusServer.stop` | 221 |
| `argus/mcp/tools.py` | 🏛️ class | `ToolAdapter` | 21 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolAdapter.to_mcp_schema` | 25 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolAdapter.create_mcp_handler` | 35 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolAdapter.create_mcp_handler.handler` | 37 |
| `argus/mcp/tools.py` | 🏛️ class | `ToolSchemaGenerator` | 48 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolSchemaGenerator.from_function` | 52 |
| `argus/mcp/tools.py` | 🏛️ class | `MCPToolWrapper` | 82 |
| `argus/mcp/tools.py` | ⚙️ method | `MCPToolWrapper.__init__` | 85 |
| `argus/mcp/tools.py` | ⚙️ method | `MCPToolWrapper.execute` | 91 |
| `argus/mcp/tools.py` | ⚙️ method | `MCPToolWrapper.get_schema` | 100 |
| `argus/mcp/tools.py` | ⚙️ method | `MCPToolWrapper.__call__` | 103 |
| `argus/mcp/tools.py` | 🏛️ class | `ToolRegistry` | 107 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolRegistry.__init__` | 110 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolRegistry.register_argus_tool` | 114 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolRegistry.register_mcp_tool` | 119 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolRegistry.get_argus_tool` | 125 |
| `argus/mcp/tools.py` | ⚙️ method | `ToolRegistry.get_mcp_wrapper` | 128 |
| `argus/memory/config.py` | 🏛️ class | `MemoryType` | 19 |
| `argus/memory/config.py` | 🏛️ class | `StorageBackend` | 30 |
| `argus/memory/config.py` | 🏛️ class | `ShortTermConfig` | 38 |
| `argus/memory/config.py` | 🏛️ class | `LongTermConfig` | 47 |
| `argus/memory/config.py` | 🏛️ class | `SemanticCacheConfig` | 57 |
| `argus/memory/config.py` | 🏛️ class | `MemoryConfig` | 66 |
| `argus/memory/config.py` | 🔧 function | `get_default_memory_config` | 76 |
| `argus/memory/long_term.py` | 🏛️ class | `MemoryEntry` | 24 |
| `argus/memory/long_term.py` | ⚙️ method | `MemoryEntry.to_dict` | 36 |
| `argus/memory/long_term.py` | ⚙️ method | `MemoryEntry.from_dict` | 43 |
| `argus/memory/long_term.py` | 🏛️ class | `BaseLongTermMemory` | 50 |
| `argus/memory/long_term.py` | ⚙️ method | `BaseLongTermMemory.add` | 54 |
| `argus/memory/long_term.py` | ⚙️ method | `BaseLongTermMemory.search` | 58 |
| `argus/memory/long_term.py` | ⚙️ method | `BaseLongTermMemory.get` | 62 |
| `argus/memory/long_term.py` | ⚙️ method | `BaseLongTermMemory.delete` | 66 |
| `argus/memory/long_term.py` | 🏛️ class | `VectorStoreMemory` | 70 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory.__init__` | 73 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory._init_index` | 86 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory._get_embedding` | 94 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory.add` | 100 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory.search` | 116 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory.get` | 146 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory.delete` | 149 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory.save` | 155 |
| `argus/memory/long_term.py` | ⚙️ method | `VectorStoreMemory._load` | 162 |
| `argus/memory/long_term.py` | 🏛️ class | `SemanticMemory` | 180 |
| `argus/memory/long_term.py` | ⚙️ method | `SemanticMemory.add_fact` | 183 |
| `argus/memory/long_term.py` | 🏛️ class | `EpisodicMemory` | 187 |
| `argus/memory/long_term.py` | ⚙️ method | `EpisodicMemory.add_episode` | 190 |
| `argus/memory/long_term.py` | 🏛️ class | `ProceduralMemory` | 194 |
| `argus/memory/long_term.py` | ⚙️ method | `ProceduralMemory.add_procedure` | 197 |
| `argus/memory/semantic_cache.py` | 🏛️ class | `CacheEntry` | 23 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `CacheEntry.is_expired` | 35 |
| `argus/memory/semantic_cache.py` | 🏛️ class | `SemanticCache` | 41 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache.__init__` | 44 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache._get_embedding` | 58 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache._generate_id` | 64 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache.get` | 67 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache.set` | 92 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache.invalidate` | 111 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache._cleanup_expired` | 121 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache._evict_lru` | 128 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache.clear` | 135 |
| `argus/memory/semantic_cache.py` | ⚙️ method | `SemanticCache.get_stats` | 141 |
| `argus/memory/short_term.py` | 🏛️ class | `Message` | 19 |
| `argus/memory/short_term.py` | ⚙️ method | `Message.to_dict` | 26 |
| `argus/memory/short_term.py` | 🏛️ class | `BaseShortTermMemory` | 31 |
| `argus/memory/short_term.py` | ⚙️ method | `BaseShortTermMemory.__init__` | 34 |
| `argus/memory/short_term.py` | ⚙️ method | `BaseShortTermMemory.add` | 39 |
| `argus/memory/short_term.py` | ⚙️ method | `BaseShortTermMemory.get_messages` | 43 |
| `argus/memory/short_term.py` | ⚙️ method | `BaseShortTermMemory.clear` | 46 |
| `argus/memory/short_term.py` | ⚙️ method | `BaseShortTermMemory.__len__` | 49 |
| `argus/memory/short_term.py` | 🏛️ class | `ConversationBufferMemory` | 53 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationBufferMemory.add` | 56 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationBufferMemory.get_messages` | 62 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationBufferMemory.get_context_string` | 65 |
| `argus/memory/short_term.py` | 🏛️ class | `ConversationWindowMemory` | 70 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationWindowMemory.__init__` | 73 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationWindowMemory.add` | 77 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationWindowMemory.get_messages` | 83 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationWindowMemory.get_full_history` | 86 |
| `argus/memory/short_term.py` | 🏛️ class | `ConversationSummaryMemory` | 90 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationSummaryMemory.__init__` | 93 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationSummaryMemory.add` | 101 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationSummaryMemory._update_summary` | 108 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationSummaryMemory.get_messages` | 123 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationSummaryMemory.get_summary` | 126 |
| `argus/memory/short_term.py` | ⚙️ method | `ConversationSummaryMemory.get_context` | 129 |
| `argus/memory/short_term.py` | 🏛️ class | `EntityMemory` | 136 |
| `argus/memory/short_term.py` | ⚙️ method | `EntityMemory.__init__` | 139 |
| `argus/memory/short_term.py` | ⚙️ method | `EntityMemory.add` | 143 |
| `argus/memory/short_term.py` | ⚙️ method | `EntityMemory.get_messages` | 153 |
| `argus/memory/short_term.py` | ⚙️ method | `EntityMemory.get_entity` | 156 |
| `argus/memory/short_term.py` | ⚙️ method | `EntityMemory.get_all_entities` | 159 |
| `argus/memory/short_term.py` | ⚙️ method | `EntityMemory.update_entity` | 162 |
| `argus/memory/short_term.py` | 🏛️ class | `ShortTermMemoryManager` | 169 |
| `argus/memory/short_term.py` | ⚙️ method | `ShortTermMemoryManager.__init__` | 172 |
| `argus/memory/short_term.py` | ⚙️ method | `ShortTermMemoryManager.add` | 185 |
| `argus/memory/short_term.py` | ⚙️ method | `ShortTermMemoryManager.get_messages` | 188 |
| `argus/memory/short_term.py` | ⚙️ method | `ShortTermMemoryManager.clear` | 191 |
| `argus/memory/store.py` | 🏛️ class | `MemoryStore` | 21 |
| `argus/memory/store.py` | ⚙️ method | `MemoryStore.save` | 25 |
| `argus/memory/store.py` | ⚙️ method | `MemoryStore.load` | 29 |
| `argus/memory/store.py` | ⚙️ method | `MemoryStore.delete` | 33 |
| `argus/memory/store.py` | ⚙️ method | `MemoryStore.list_keys` | 37 |
| `argus/memory/store.py` | ⚙️ method | `MemoryStore.clear` | 41 |
| `argus/memory/store.py` | 🏛️ class | `InMemoryStore` | 45 |
| `argus/memory/store.py` | ⚙️ method | `InMemoryStore.__init__` | 48 |
| `argus/memory/store.py` | ⚙️ method | `InMemoryStore.save` | 51 |
| `argus/memory/store.py` | ⚙️ method | `InMemoryStore.load` | 55 |
| `argus/memory/store.py` | ⚙️ method | `InMemoryStore.delete` | 58 |
| `argus/memory/store.py` | ⚙️ method | `InMemoryStore.list_keys` | 64 |
| `argus/memory/store.py` | ⚙️ method | `InMemoryStore.clear` | 67 |
| `argus/memory/store.py` | 🏛️ class | `SQLiteStore` | 73 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore.__init__` | 76 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore._init_db` | 81 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore.save` | 92 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore.load` | 105 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore.delete` | 112 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore.list_keys` | 117 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore.clear` | 121 |
| `argus/memory/store.py` | ⚙️ method | `SQLiteStore.close` | 126 |
| `argus/memory/store.py` | 🏛️ class | `FAISSStore` | 130 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore.__init__` | 133 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore._init_index` | 142 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore.save` | 149 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore.load` | 160 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore.delete` | 168 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore.list_keys` | 178 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore.clear` | 181 |
| `argus/memory/store.py` | ⚙️ method | `FAISSStore.search` | 189 |
| `argus/memory/store.py` | 🏛️ class | `FileSystemStore` | 201 |
| `argus/memory/store.py` | ⚙️ method | `FileSystemStore.__init__` | 204 |
| `argus/memory/store.py` | ⚙️ method | `FileSystemStore._get_path` | 208 |
| `argus/memory/store.py` | ⚙️ method | `FileSystemStore.save` | 212 |
| `argus/memory/store.py` | ⚙️ method | `FileSystemStore.load` | 221 |
| `argus/memory/store.py` | ⚙️ method | `FileSystemStore.delete` | 227 |
| `argus/memory/store.py` | ⚙️ method | `FileSystemStore.list_keys` | 234 |
| `argus/memory/store.py` | ⚙️ method | `FileSystemStore.clear` | 237 |
| `argus/metrics/collector.py` | 🏛️ class | `MetricType` | 35 |
| `argus/metrics/collector.py` | 🏛️ class | `Metric` | 43 |
| `argus/metrics/collector.py` | ⚙️ method | `Metric.to_dict` | 51 |
| `argus/metrics/collector.py` | 🏛️ class | `MetricsCollector` | 61 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.__init__` | 79 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.set_default_tags` | 99 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.record` | 108 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.record_counter` | 151 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.record_gauge` | 166 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.record_histogram` | 181 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.get_counter` | 196 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.get_gauge` | 208 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.get_histogram_stats` | 220 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.get_stats` | 249 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.get_history` | 266 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.reset` | 296 |
| `argus/metrics/collector.py` | ⚙️ method | `MetricsCollector.export` | 305 |
| `argus/metrics/collector.py` | 🔧 function | `get_default_collector` | 323 |
| `argus/metrics/collector.py` | 🔧 function | `record_metric` | 337 |
| `argus/metrics/collector.py` | 🔧 function | `record_counter` | 347 |
| `argus/metrics/collector.py` | 🔧 function | `record_gauge` | 356 |
| `argus/metrics/collector.py` | 🔧 function | `record_histogram` | 365 |
| `argus/metrics/traces.py` | 🏛️ class | `SpanStatus` | 41 |
| `argus/metrics/traces.py` | 🏛️ class | `TraceConfig` | 48 |
| `argus/metrics/traces.py` | 🏛️ class | `SpanContext` | 78 |
| `argus/metrics/traces.py` | ⚙️ method | `SpanContext.to_dict` | 84 |
| `argus/metrics/traces.py` | 🏛️ class | `Span` | 93 |
| `argus/metrics/traces.py` | ⚙️ method | `Span.set_attribute` | 107 |
| `argus/metrics/traces.py` | ⚙️ method | `Span.set_attributes` | 120 |
| `argus/metrics/traces.py` | ⚙️ method | `Span.add_event` | 132 |
| `argus/metrics/traces.py` | ⚙️ method | `Span.set_status` | 153 |
| `argus/metrics/traces.py` | ⚙️ method | `Span.end` | 168 |
| `argus/metrics/traces.py` | ⚙️ method | `Span.duration_ms` | 175 |
| `argus/metrics/traces.py` | ⚙️ method | `Span.to_dict` | 183 |
| `argus/metrics/traces.py` | 🏛️ class | `Tracer` | 197 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.__init__` | 214 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer._generate_id` | 233 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer._should_sample` | 237 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.span` | 247 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.start_span` | 317 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.get_current_span` | 351 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.get_trace` | 359 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.get_recent_spans` | 371 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.get_stats` | 383 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.export` | 400 |
| `argus/metrics/traces.py` | ⚙️ method | `Tracer.clear` | 409 |
| `argus/metrics/traces.py` | 🏛️ class | `_NoOpSpan` | 416 |
| `argus/metrics/traces.py` | ⚙️ method | `_NoOpSpan.set_attribute` | 419 |
| `argus/metrics/traces.py` | ⚙️ method | `_NoOpSpan.set_attributes` | 422 |
| `argus/metrics/traces.py` | ⚙️ method | `_NoOpSpan.add_event` | 425 |
| `argus/metrics/traces.py` | ⚙️ method | `_NoOpSpan.set_status` | 428 |
| `argus/metrics/traces.py` | 🔧 function | `get_tracer` | 440 |
| `argus/orchestrator.py` | 🏛️ class | `DebateResult` | 36 |
| `argus/orchestrator.py` | ⚙️ method | `DebateResult.to_dict` | 57 |
| `argus/orchestrator.py` | 🏛️ class | `RDCOrchestrator` | 69 |
| `argus/orchestrator.py` | ⚙️ method | `RDCOrchestrator.__init__` | 85 |
| `argus/orchestrator.py` | ⚙️ method | `RDCOrchestrator.debate` | 113 |
| `argus/orchestrator.py` | ⚙️ method | `RDCOrchestrator.quick_evaluate` | 231 |
| `argus/orchestrator.py` | ⚙️ method | `RDCOrchestrator.graph` | 286 |
| `argus/orchestrator.py` | ⚙️ method | `RDCOrchestrator.agenda` | 291 |
| `argus/outputs/plotting.py` | 🏛️ class | `PlotTheme` | 60 |
| `argus/outputs/plotting.py` | 🏛️ class | `PlotConfig` | 122 |
| `argus/outputs/plotting.py` | ⚙️ method | `PlotConfig.__post_init__` | 135 |
| `argus/outputs/plotting.py` | 🔧 function | `setup_plot_style` | 140 |
| `argus/outputs/plotting.py` | 🔧 function | `extract_posterior_timeline` | 197 |
| `argus/outputs/plotting.py` | 🔧 function | `extract_evidence_by_polarity` | 211 |
| `argus/outputs/plotting.py` | 🔧 function | `extract_specialist_contributions` | 226 |
| `argus/outputs/plotting.py` | 🔧 function | `extract_evidence_confidence` | 243 |
| `argus/outputs/plotting.py` | 🔧 function | `extract_graph_structure` | 255 |
| `argus/outputs/plotting.py` | 🏛️ class | `DebatePlotter` | 265 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.__init__` | 279 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter._save_figure` | 285 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_posterior_evolution` | 298 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_evidence_distribution` | 372 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_specialist_contributions` | 449 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_confidence_distribution` | 515 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_round_heatmap` | 584 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_cdag_network` | 647 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_multi_stock_comparison` | 760 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.plot_summary_radar` | 849 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.generate_all_plots` | 903 |
| `argus/outputs/plotting.py` | ⚙️ method | `DebatePlotter.generate_all_comparison_plots` | 935 |
| `argus/outputs/plotting.py` | 🏛️ class | `InteractivePlotter` | 969 |
| `argus/outputs/plotting.py` | ⚙️ method | `InteractivePlotter.__init__` | 974 |
| `argus/outputs/plotting.py` | ⚙️ method | `InteractivePlotter._save_html` | 980 |
| `argus/outputs/plotting.py` | ⚙️ method | `InteractivePlotter.plot_interactive_posterior` | 987 |
| `argus/outputs/plotting.py` | ⚙️ method | `InteractivePlotter.plot_interactive_network` | 1027 |
| `argus/outputs/plotting.py` | ⚙️ method | `InteractivePlotter.plot_dashboard` | 1112 |
| `argus/outputs/plotting.py` | 🔧 function | `generate_debate_plots` | 1189 |
| `argus/outputs/plotting.py` | 🔧 function | `generate_comparison_plots` | 1218 |
| `argus/outputs/reports.py` | 🏛️ class | `ReportFormat` | 41 |
| `argus/outputs/reports.py` | 🏛️ class | `ReportConfig` | 49 |
| `argus/outputs/reports.py` | 🏛️ class | `EvidenceSummary` | 86 |
| `argus/outputs/reports.py` | ⚙️ method | `EvidenceSummary.to_dict` | 95 |
| `argus/outputs/reports.py` | 🏛️ class | `VerdictSummary` | 100 |
| `argus/outputs/reports.py` | ⚙️ method | `VerdictSummary.to_dict` | 108 |
| `argus/outputs/reports.py` | 🏛️ class | `DebateReport` | 113 |
| `argus/outputs/reports.py` | ⚙️ method | `DebateReport.to_dict` | 145 |
| `argus/outputs/reports.py` | ⚙️ method | `DebateReport.to_json` | 176 |
| `argus/outputs/reports.py` | ⚙️ method | `DebateReport.to_markdown` | 187 |
| `argus/outputs/reports.py` | ⚙️ method | `DebateReport.to_summary` | 257 |
| `argus/outputs/reports.py` | 🏛️ class | `ReportGenerator` | 274 |
| `argus/outputs/reports.py` | ⚙️ method | `ReportGenerator.__init__` | 290 |
| `argus/outputs/reports.py` | ⚙️ method | `ReportGenerator.generate` | 299 |
| `argus/outputs/reports.py` | ⚙️ method | `ReportGenerator._extract_graph_data` | 386 |
| `argus/outputs/reports.py` | ⚙️ method | `ReportGenerator.generate_batch` | 432 |
| `argus/outputs/reports.py` | 🔧 function | `generate_report` | 451 |
| `argus/outputs/reports.py` | 🔧 function | `export_json` | 468 |
| `argus/outputs/reports.py` | 🔧 function | `export_markdown` | 494 |
| `argus/provenance/integrity.py` | 🔧 function | `compute_hash` | 17 |
| `argus/provenance/integrity.py` | 🔧 function | `verify_hash` | 38 |
| `argus/provenance/integrity.py` | 🔧 function | `compute_merkle_root` | 54 |
| `argus/provenance/integrity.py` | 🏛️ class | `Attestation` | 84 |
| `argus/provenance/integrity.py` | ⚙️ method | `Attestation.__post_init__` | 103 |
| `argus/provenance/integrity.py` | ⚙️ method | `Attestation.to_dict` | 107 |
| `argus/provenance/integrity.py` | ⚙️ method | `Attestation.verify` | 118 |
| `argus/provenance/integrity.py` | 🔧 function | `create_attestation` | 131 |
| `argus/provenance/integrity.py` | 🔧 function | `create_batch_attestation` | 160 |
| `argus/provenance/integrity.py` | 🏛️ class | `IntegrityChecker` | 193 |
| `argus/provenance/integrity.py` | ⚙️ method | `IntegrityChecker.__init__` | 200 |
| `argus/provenance/integrity.py` | ⚙️ method | `IntegrityChecker.register` | 203 |
| `argus/provenance/integrity.py` | ⚙️ method | `IntegrityChecker.verify` | 224 |
| `argus/provenance/integrity.py` | ⚙️ method | `IntegrityChecker.get_attestation` | 248 |
| `argus/provenance/ledger.py` | 🏛️ class | `EventType` | 24 |
| `argus/provenance/ledger.py` | 🏛️ class | `ProvenanceEvent` | 55 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceEvent.to_dict` | 82 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceEvent.to_prov` | 96 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceEvent.from_dict` | 113 |
| `argus/provenance/ledger.py` | 🏛️ class | `ProvenanceLedger` | 128 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger.__init__` | 145 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger._generate_id` | 168 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger._compute_hash` | 174 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger.record` | 189 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger.query` | 245 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger.get_entity_history` | 291 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger.verify_integrity` | 306 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger._append_to_file` | 328 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger._load` | 333 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger.export_prov` | 345 |
| `argus/provenance/ledger.py` | ⚙️ method | `ProvenanceLedger.__len__` | 360 |
| `argus/retrieval/cite_critique.py` | 🏛️ class | `CiteResult` | 22 |
| `argus/retrieval/cite_critique.py` | ⚙️ method | `CiteResult.to_dict` | 43 |
| `argus/retrieval/cite_critique.py` | 🔧 function | `cite_and_critique` | 56 |
| `argus/retrieval/cite_critique.py` | 🔧 function | `extract_claims` | 133 |
| `argus/retrieval/cite_critique.py` | 🔧 function | `build_evidence_summary` | 169 |
| `argus/retrieval/hybrid.py` | 🏛️ class | `RetrievalResult` | 25 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `RetrievalResult.to_dict` | 46 |
| `argus/retrieval/hybrid.py` | 🏛️ class | `HybridRetriever` | 59 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever.__init__` | 77 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever._init_components` | 102 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever._init_reranker` | 111 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever.index_chunks` | 122 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever.retrieve` | 147 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever._rerank` | 195 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever.retrieve_with_rrf` | 225 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever.clear` | 288 |
| `argus/retrieval/hybrid.py` | ⚙️ method | `HybridRetriever.num_indexed` | 294 |
| `argus/retrieval/reranker.py` | 🏛️ class | `RerankResult` | 17 |
| `argus/retrieval/reranker.py` | 🏛️ class | `CrossEncoderReranker` | 26 |
| `argus/retrieval/reranker.py` | ⚙️ method | `CrossEncoderReranker.__init__` | 38 |
| `argus/retrieval/reranker.py` | ⚙️ method | `CrossEncoderReranker._load_model` | 58 |
| `argus/retrieval/reranker.py` | ⚙️ method | `CrossEncoderReranker.score_pairs` | 76 |
| `argus/retrieval/reranker.py` | ⚙️ method | `CrossEncoderReranker.rerank` | 98 |
| `argus/retrieval/reranker.py` | 🔧 function | `rerank_results` | 145 |
| `argus/tools/base.py` | 🏛️ class | `ToolCategory` | 40 |
| `argus/tools/base.py` | 🏛️ class | `ToolConfig` | 68 |
| `argus/tools/base.py` | 🏛️ class | `ToolResult` | 100 |
| `argus/tools/base.py` | ⚙️ method | `ToolResult.to_dict` | 119 |
| `argus/tools/base.py` | ⚙️ method | `ToolResult.from_error` | 132 |
| `argus/tools/base.py` | ⚙️ method | `ToolResult.from_data` | 137 |
| `argus/tools/base.py` | 🏛️ class | `BaseTool` | 142 |
| `argus/tools/base.py` | ⚙️ method | `BaseTool.__init__` | 171 |
| `argus/tools/base.py` | ⚙️ method | `BaseTool.execute` | 183 |
| `argus/tools/base.py` | ⚙️ method | `BaseTool.validate_input` | 196 |
| `argus/tools/base.py` | ⚙️ method | `BaseTool.__call__` | 209 |
| `argus/tools/base.py` | ⚙️ method | `BaseTool.get_schema` | 247 |
| `argus/tools/base.py` | ⚙️ method | `BaseTool.get_stats` | 263 |
| `argus/tools/base.py` | ⚙️ method | `BaseTool.__repr__` | 281 |
| `argus/tools/base.py` | 🏛️ class | `EchoTool` | 289 |
| `argus/tools/base.py` | ⚙️ method | `EchoTool.execute` | 298 |
| `argus/tools/base.py` | ⚙️ method | `EchoTool.validate_input` | 312 |
| `argus/tools/base.py` | 🏛️ class | `CalculatorTool` | 318 |
| `argus/tools/base.py` | ⚙️ method | `CalculatorTool.execute` | 337 |
| `argus/tools/base.py` | ⚙️ method | `CalculatorTool.validate_input` | 398 |
| `argus/tools/base.py` | 🔧 function | `create_tool` | 405 |
| `argus/tools/cache.py` | 🏛️ class | `CacheConfig` | 39 |
| `argus/tools/cache.py` | 🏛️ class | `CacheEntry` | 65 |
| `argus/tools/cache.py` | 🏛️ class | `ResultCache` | 73 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache.__init__` | 89 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache._compute_key` | 106 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache.get` | 123 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache.set` | 162 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache.invalidate` | 195 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache.clear` | 234 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache._maybe_cleanup` | 240 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache._cleanup_expired` | 249 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache.get_stats` | 269 |
| `argus/tools/cache.py` | ⚙️ method | `ResultCache.__len__` | 287 |
| `argus/tools/cache.py` | 🔧 function | `cached_tool` | 297 |
| `argus/tools/cache.py` | ⚙️ method | `cached_tool.decorator` | 318 |
| `argus/tools/cache.py` | ⚙️ method | `cached_tool.decorator.wrapper` | 320 |
| `argus/tools/executor.py` | 🏛️ class | `ExecutorConfig` | 39 |
| `argus/tools/executor.py` | 🏛️ class | `ExecutionRecord` | 70 |
| `argus/tools/executor.py` | 🏛️ class | `ToolExecutor` | 81 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor.__init__` | 106 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor.run` | 130 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor._execute_with_timeout` | 213 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor.run_batch` | 238 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor._record_execution` | 261 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor.get_history` | 288 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor.get_stats` | 307 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor.add_guardrail` | 329 |
| `argus/tools/executor.py` | ⚙️ method | `ToolExecutor.set_cache` | 338 |
| `argus/tools/guardrails.py` | 🏛️ class | `GuardrailConfig` | 33 |
| `argus/tools/guardrails.py` | 🏛️ class | `GuardrailResult` | 56 |
| `argus/tools/guardrails.py` | 🏛️ class | `Guardrail` | 69 |
| `argus/tools/guardrails.py` | ⚙️ method | `Guardrail.__init__` | 87 |
| `argus/tools/guardrails.py` | ⚙️ method | `Guardrail.check` | 96 |
| `argus/tools/guardrails.py` | ⚙️ method | `Guardrail.check_output` | 112 |
| `argus/tools/guardrails.py` | 🏛️ class | `ContentFilter` | 135 |
| `argus/tools/guardrails.py` | ⚙️ method | `ContentFilter.__init__` | 150 |
| `argus/tools/guardrails.py` | ⚙️ method | `ContentFilter.check` | 176 |
| `argus/tools/guardrails.py` | ⚙️ method | `ContentFilter._arguments_to_text` | 213 |
| `argus/tools/guardrails.py` | 🏛️ class | `PolicyEnforcer` | 221 |
| `argus/tools/guardrails.py` | ⚙️ method | `PolicyEnforcer.__init__` | 236 |
| `argus/tools/guardrails.py` | ⚙️ method | `PolicyEnforcer.check` | 257 |
| `argus/tools/guardrails.py` | ⚙️ method | `PolicyEnforcer.reset_counts` | 298 |
| `argus/tools/guardrails.py` | ⚙️ method | `PolicyEnforcer.get_counts` | 302 |
| `argus/tools/guardrails.py` | 🏛️ class | `RateLimiter` | 307 |
| `argus/tools/guardrails.py` | ⚙️ method | `RateLimiter.__init__` | 322 |
| `argus/tools/guardrails.py` | ⚙️ method | `RateLimiter.check` | 341 |
| `argus/tools/guardrails.py` | 🏛️ class | `InputValidator` | 389 |
| `argus/tools/guardrails.py` | ⚙️ method | `InputValidator.__init__` | 402 |
| `argus/tools/guardrails.py` | ⚙️ method | `InputValidator.check` | 419 |
| `argus/tools/integrations/__init__.py` | 🔧 function | `list_all_tools` | 253 |
| `argus/tools/integrations/__init__.py` | 🔧 function | `list_tool_categories` | 258 |
| `argus/tools/integrations/__init__.py` | 🔧 function | `get_tools_by_category` | 263 |
| `argus/tools/integrations/__init__.py` | 🔧 function | `get_all_tools` | 270 |
| `argus/tools/integrations/__init__.py` | 🔧 function | `get_tool_count` | 279 |
| `argus/tools/integrations/ai_agents/agentmail.py` | 🏛️ class | `EmailMessage` | 28 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `EmailMessage.to_dict` | 39 |
| `argus/tools/integrations/ai_agents/agentmail.py` | 🏛️ class | `AgentInbox` | 53 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentInbox.add_message` | 61 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentInbox.get_unread` | 64 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentInbox.mark_read` | 67 |
| `argus/tools/integrations/ai_agents/agentmail.py` | 🏛️ class | `AgentMailTool` | 75 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool.__init__` | 95 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool.execute` | 127 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._create_inbox` | 188 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._delete_inbox` | 216 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._list_inboxes` | 228 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._send_email` | 243 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._receive_emails` | 294 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._get_message` | 360 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._mark_read` | 379 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._search_messages` | 396 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool._get_stats` | 424 |
| `argus/tools/integrations/ai_agents/agentmail.py` | ⚙️ method | `AgentMailTool.get_schema` | 451 |
| `argus/tools/integrations/ai_agents/agentops.py` | 🏛️ class | `AgentEvent` | 26 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentEvent.to_dict` | 38 |
| `argus/tools/integrations/ai_agents/agentops.py` | 🏛️ class | `AgentSession` | 53 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentSession.is_active` | 65 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentSession.duration_seconds` | 69 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentSession.add_event` | 73 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentSession.to_dict` | 76 |
| `argus/tools/integrations/ai_agents/agentops.py` | 🏛️ class | `AgentOpsTool` | 91 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool.__init__` | 116 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool.execute` | 136 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._register_agent` | 200 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._start_session` | 228 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._end_session` | 273 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._track_event_internal` | 303 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._track_event` | 327 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._get_replay` | 370 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._list_sessions` | 397 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._get_session` | 424 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._get_metrics` | 436 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._get_agent_stats` | 457 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool._search_events` | 488 |
| `argus/tools/integrations/ai_agents/agentops.py` | ⚙️ method | `AgentOpsTool.get_schema` | 513 |
| `argus/tools/integrations/ai_agents/freeplay.py` | 🏛️ class | `PromptTemplate` | 24 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `PromptTemplate.render` | 34 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `PromptTemplate.to_dict` | 42 |
| `argus/tools/integrations/ai_agents/freeplay.py` | 🏛️ class | `Experiment` | 55 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `Experiment.add_result` | 64 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `Experiment.get_winner` | 67 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `Experiment.to_dict` | 75 |
| `argus/tools/integrations/ai_agents/freeplay.py` | 🏛️ class | `EvaluationRun` | 88 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `EvaluationRun.to_dict` | 99 |
| `argus/tools/integrations/ai_agents/freeplay.py` | 🏛️ class | `FreeplayTool` | 112 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool.__init__` | 137 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool.execute` | 156 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._create_template` | 219 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._update_template` | 252 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._render_template` | 280 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._list_templates` | 299 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._get_template` | 304 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._create_experiment` | 315 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._record_result` | 349 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._get_experiment` | 374 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._list_experiments` | 385 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._create_evaluation` | 390 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._run_evaluation` | 416 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._get_evaluation` | 459 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._list_evaluations` | 470 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._log_trace` | 475 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._get_traces` | 494 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool._get_analytics` | 503 |
| `argus/tools/integrations/ai_agents/freeplay.py` | ⚙️ method | `FreeplayTool.get_schema` | 514 |
| `argus/tools/integrations/ai_agents/goodmem.py` | 🏛️ class | `MemoryEntry` | 26 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `MemoryEntry.to_dict` | 39 |
| `argus/tools/integrations/ai_agents/goodmem.py` | 🏛️ class | `MemorySpace` | 54 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `MemorySpace.add_memory` | 62 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `MemorySpace.to_dict` | 65 |
| `argus/tools/integrations/ai_agents/goodmem.py` | 🏛️ class | `GoodMemTool` | 75 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool.__init__` | 99 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._get_embedder` | 124 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._compute_embedding` | 135 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._cosine_similarity` | 144 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._load_memories` | 154 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._save_memories` | 192 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool.execute` | 216 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._get_or_create_space` | 292 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._store_memory` | 306 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._search_memories` | 348 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._recall_memory` | 404 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._forget_memory` | 426 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._list_memories` | 447 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._create_space` | 473 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._list_spaces` | 491 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._consolidate_memories` | 508 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._get_stats` | 559 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool._update_importance` | 589 |
| `argus/tools/integrations/ai_agents/goodmem.py` | ⚙️ method | `GoodMemTool.get_schema` | 610 |
| `argus/tools/integrations/cloud/bigquery.py` | 🏛️ class | `QueryResult` | 23 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `QueryResult.to_dict` | 33 |
| `argus/tools/integrations/cloud/bigquery.py` | 🏛️ class | `BigQueryTool` | 45 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool.__init__` | 67 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._get_client` | 83 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool.execute` | 93 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._execute_query` | 152 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._list_datasets` | 238 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._list_tables` | 256 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._get_schema` | 281 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._preview_table` | 313 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._create_dataset` | 327 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._delete_dataset` | 354 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._analyze_table` | 373 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool._estimate_cost` | 416 |
| `argus/tools/integrations/cloud/bigquery.py` | ⚙️ method | `BigQueryTool.get_schema` | 424 |
| `argus/tools/integrations/cloud/cloud_trace.py` | 🏛️ class | `Span` | 25 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `Span.duration_ms` | 38 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `Span.add_event` | 43 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `Span.to_dict` | 50 |
| `argus/tools/integrations/cloud/cloud_trace.py` | 🏛️ class | `Trace` | 66 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `Trace.root_span` | 74 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `Trace.duration_ms` | 81 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `Trace.to_dict` | 86 |
| `argus/tools/integrations/cloud/cloud_trace.py` | 🏛️ class | `CloudTraceTool` | 97 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool.__init__` | 121 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._get_trace_client` | 142 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool.execute` | 152 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._start_trace` | 200 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._end_trace` | 238 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._start_span` | 269 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._end_span` | 308 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._add_event` | 336 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._set_status` | 358 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._set_attributes` | 378 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._get_trace` | 396 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._get_span` | 408 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._list_traces` | 420 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._export_trace` | 442 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._export_to_cloud` | 471 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool._get_active_span` | 497 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool.trace_context` | 513 |
| `argus/tools/integrations/cloud/cloud_trace.py` | ⚙️ method | `CloudTraceTool.get_schema` | 529 |
| `argus/tools/integrations/cloud/pubsub.py` | 🏛️ class | `PubSubTool` | 21 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool.__init__` | 42 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._get_publisher` | 57 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._get_subscriber` | 67 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool.execute` | 77 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._topic_path` | 142 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._subscription_path` | 146 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._publish` | 150 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._publish_batch` | 186 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._pull` | 226 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._acknowledge` | 265 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._list_topics` | 292 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._list_subscriptions` | 309 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._create_topic` | 334 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._delete_topic` | 354 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._create_subscription` | 373 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._delete_subscription` | 405 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._get_topic` | 424 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool._get_subscription` | 443 |
| `argus/tools/integrations/cloud/pubsub.py` | ⚙️ method | `PubSubTool.get_schema` | 465 |
| `argus/tools/integrations/cloud/vertex_ai.py` | 🏛️ class | `VertexAISearchTool` | 21 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool.__init__` | 41 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool._get_client` | 59 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool.execute` | 69 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool._search` | 109 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool._list_datastores` | 163 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool._get_datastore` | 187 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool._get_document` | 213 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAISearchTool.get_schema` | 244 |
| `argus/tools/integrations/cloud/vertex_ai.py` | 🏛️ class | `VertexAIRAGTool` | 265 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool.__init__` | 285 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool.execute` | 298 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool._list_corpora` | 338 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool._create_corpus` | 362 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool._delete_corpus` | 388 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool._retrieve` | 409 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool._import_files` | 447 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool._list_files` | 480 |
| `argus/tools/integrations/cloud/vertex_ai.py` | ⚙️ method | `VertexAIRAGTool.get_schema` | 510 |
| `argus/tools/integrations/communication/mailgun.py` | 🏛️ class | `MailgunTool` | 19 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool.__init__` | 42 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._get_session` | 60 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._request` | 74 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool.execute` | 103 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._send_email` | 186 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._send_template` | 239 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._list_mailing_lists` | 279 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._create_mailing_list` | 302 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._get_mailing_list` | 330 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._delete_mailing_list` | 343 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._add_member` | 359 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._list_members` | 391 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._remove_member` | 422 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._get_events` | 443 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._list_domains` | 481 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._get_domain` | 504 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._verify_domain` | 518 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool._get_stats` | 532 |
| `argus/tools/integrations/communication/mailgun.py` | ⚙️ method | `MailgunTool.get_schema` | 554 |
| `argus/tools/integrations/communication/paypal.py` | 🏛️ class | `PayPalTool` | 19 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool.__init__` | 44 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_access_token` | 67 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_session` | 93 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._request` | 106 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool.execute` | 137 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._create_order` | 215 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_order` | 286 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._capture_order` | 307 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._authorize_order` | 331 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._refund_capture` | 349 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_capture` | 379 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._create_payout` | 393 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_payout` | 436 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._list_products` | 456 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._create_product` | 475 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_product` | 501 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._list_plans` | 512 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._create_plan` | 531 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_plan` | 588 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._create_subscription` | 602 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._get_subscription` | 637 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool._cancel_subscription` | 658 |
| `argus/tools/integrations/communication/paypal.py` | ⚙️ method | `PayPalTool.get_schema` | 680 |
| `argus/tools/integrations/communication/stripe_tool.py` | 🏛️ class | `StripeTool` | 18 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool.__init__` | 43 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_session` | 59 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._request` | 70 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool.execute` | 98 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._list_customers` | 192 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_customer` | 216 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._create_customer` | 239 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._update_customer` | 268 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._delete_customer` | 298 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._list_payment_intents` | 315 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_payment_intent` | 342 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._create_payment_intent` | 365 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._confirm_payment_intent` | 400 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._cancel_payment_intent` | 426 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._list_subscriptions` | 447 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_subscription` | 477 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._create_subscription` | 499 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._update_subscription` | 524 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._cancel_subscription` | 554 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._list_invoices` | 572 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_invoice` | 603 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._create_invoice` | 616 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._finalize_invoice` | 635 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._pay_invoice` | 652 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._list_products` | 670 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_product` | 693 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._create_product` | 706 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._list_prices` | 730 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_price` | 761 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._create_price` | 774 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool._get_balance` | 805 |
| `argus/tools/integrations/communication/stripe_tool.py` | ⚙️ method | `StripeTool.get_schema` | 824 |
| `argus/tools/integrations/database/pandas_tool.py` | 🏛️ class | `PandasTool` | 17 |
| `argus/tools/integrations/database/pandas_tool.py` | ⚙️ method | `PandasTool.__init__` | 31 |
| `argus/tools/integrations/database/pandas_tool.py` | ⚙️ method | `PandasTool.execute` | 35 |
| `argus/tools/integrations/database/pandas_tool.py` | ⚙️ method | `PandasTool.get_schema` | 134 |
| `argus/tools/integrations/database/sql.py` | 🏛️ class | `SqlTool` | 17 |
| `argus/tools/integrations/database/sql.py` | ⚙️ method | `SqlTool.__init__` | 33 |
| `argus/tools/integrations/database/sql.py` | ⚙️ method | `SqlTool._get_engine` | 44 |
| `argus/tools/integrations/database/sql.py` | ⚙️ method | `SqlTool._is_read_query` | 56 |
| `argus/tools/integrations/database/sql.py` | ⚙️ method | `SqlTool.execute` | 61 |
| `argus/tools/integrations/database/sql.py` | ⚙️ method | `SqlTool.get_schema` | 119 |
| `argus/tools/integrations/devops/daytona.py` | 🏛️ class | `DaytonaTool` | 18 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool.__init__` | 40 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._get_session` | 58 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._request` | 71 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool.execute` | 101 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._list_workspaces` | 169 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._get_workspace` | 190 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._create_workspace` | 212 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._delete_workspace` | 243 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._start_workspace` | 262 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._stop_workspace` | 278 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._list_projects` | 295 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._get_project` | 314 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._create_project` | 332 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._delete_project` | 362 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._list_providers` | 383 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._get_provider` | 403 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._add_provider` | 418 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._remove_provider` | 451 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._list_targets` | 468 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._get_target` | 487 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._set_target` | 502 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._remove_target` | 531 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._server_info` | 548 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool._get_ssh_key` | 562 |
| `argus/tools/integrations/devops/daytona.py` | ⚙️ method | `DaytonaTool.get_schema` | 570 |
| `argus/tools/integrations/devops/gitlab.py` | 🏛️ class | `GitLabTool` | 19 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool.__init__` | 41 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._get_session` | 59 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._request` | 70 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._encode_project_path` | 100 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool.execute` | 106 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._list_projects` | 206 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._get_project` | 237 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._create_project` | 262 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._delete_project` | 289 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._list_issues` | 307 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._get_issue` | 348 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._create_issue` | 377 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._update_issue` | 411 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._close_issue` | 446 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._list_merge_requests` | 472 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._get_merge_request` | 510 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._create_merge_request` | 539 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._merge_merge_request` | 577 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._close_merge_request` | 608 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._list_pipelines` | 634 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._get_pipeline` | 668 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._create_pipeline` | 695 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._cancel_pipeline` | 721 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._retry_pipeline` | 742 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._list_jobs` | 764 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._get_job` | 801 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._retry_job` | 829 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._cancel_job` | 850 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._list_branches` | 872 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._get_branch` | 905 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._create_branch` | 930 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._delete_branch` | 956 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._search` | 978 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool._me` | 1003 |
| `argus/tools/integrations/devops/gitlab.py` | ⚙️ method | `GitLabTool.get_schema` | 1017 |
| `argus/tools/integrations/devops/n8n.py` | 🏛️ class | `N8nTool` | 18 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool.__init__` | 40 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._get_session` | 58 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._request` | 69 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool.execute` | 99 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._list_workflows` | 169 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._get_workflow` | 205 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._create_workflow` | 229 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._update_workflow` | 266 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._delete_workflow` | 293 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._activate_workflow` | 309 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._deactivate_workflow` | 326 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._execute_workflow` | 343 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._list_executions` | 371 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._get_execution` | 408 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._delete_execution` | 435 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._list_credentials` | 452 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._get_credential` | 479 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._create_credential` | 500 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._delete_credential` | 529 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._list_tags` | 546 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._get_tag` | 572 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._create_tag` | 592 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._update_tag` | 609 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._delete_tag` | 629 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool._test_webhook` | 646 |
| `argus/tools/integrations/devops/n8n.py` | ⚙️ method | `N8nTool.get_schema` | 674 |
| `argus/tools/integrations/devops/postman.py` | 🏛️ class | `PostmanTool` | 18 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool.__init__` | 42 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._get_session` | 58 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._request` | 69 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool.execute` | 98 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._list_collections` | 179 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._get_collection` | 207 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._create_collection` | 232 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._update_collection` | 276 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._delete_collection` | 307 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._fork_collection` | 323 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._list_environments` | 351 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._get_environment` | 378 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._create_environment` | 399 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._update_environment` | 438 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._delete_environment` | 471 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._list_workspaces` | 488 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._get_workspace` | 513 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._create_workspace` | 539 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._list_monitors` | 572 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._get_monitor` | 599 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._run_monitor` | 623 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._list_mocks` | 644 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._get_mock` | 672 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._create_mock` | 696 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._list_apis` | 740 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._get_api` | 767 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._create_api` | 791 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool._me` | 831 |
| `argus/tools/integrations/devops/postman.py` | ⚙️ method | `PostmanTool.get_schema` | 846 |
| `argus/tools/integrations/finance/weather.py` | 🏛️ class | `WeatherTool` | 18 |
| `argus/tools/integrations/finance/weather.py` | ⚙️ method | `WeatherTool.__init__` | 33 |
| `argus/tools/integrations/finance/weather.py` | ⚙️ method | `WeatherTool.execute` | 41 |
| `argus/tools/integrations/finance/weather.py` | ⚙️ method | `WeatherTool.get_schema` | 120 |
| `argus/tools/integrations/finance/yahoo_finance.py` | 🏛️ class | `YahooFinanceTool` | 17 |
| `argus/tools/integrations/finance/yahoo_finance.py` | ⚙️ method | `YahooFinanceTool.__init__` | 30 |
| `argus/tools/integrations/finance/yahoo_finance.py` | ⚙️ method | `YahooFinanceTool.execute` | 33 |
| `argus/tools/integrations/finance/yahoo_finance.py` | ⚙️ method | `YahooFinanceTool.get_schema` | 128 |
| `argus/tools/integrations/media_ai/cartesia.py` | 🏛️ class | `CartesiaTool` | 19 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool.__init__` | 43 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._get_session` | 59 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._request` | 71 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool.execute` | 110 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._list_voices` | 163 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._get_voice` | 183 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._create_voice` | 206 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._clone_voice` | 240 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._update_voice` | 292 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._delete_voice` | 321 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._text_to_speech` | 338 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._text_to_speech_bytes` | 395 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._create_embedding` | 411 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool._list_languages` | 436 |
| `argus/tools/integrations/media_ai/cartesia.py` | ⚙️ method | `CartesiaTool.get_schema` | 462 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | 🏛️ class | `ElevenLabsTool` | 19 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool.__init__` | 44 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._get_session` | 60 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._request` | 71 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool.execute` | 112 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._list_voices` | 175 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._get_voice` | 194 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._add_voice` | 226 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._edit_voice` | 284 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._delete_voice` | 323 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._get_voice_settings` | 339 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._text_to_speech` | 360 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._text_to_speech_stream` | 408 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._list_models` | 425 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._list_history` | 447 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._get_history_item` | 485 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._delete_history_item` | 507 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._download_history_audio` | 523 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._generate_sound_effect` | 547 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._get_user` | 583 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool._get_subscription` | 596 |
| `argus/tools/integrations/media_ai/elevenlabs.py` | ⚙️ method | `ElevenLabsTool.get_schema` | 613 |
| `argus/tools/integrations/media_ai/huggingface.py` | 🏛️ class | `HuggingFaceTool` | 19 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool.__init__` | 44 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._get_session` | 60 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._request` | 71 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool.execute` | 110 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._search_models` | 182 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._get_model` | 224 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._list_model_files` | 249 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._search_datasets` | 275 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._get_dataset` | 310 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._search_spaces` | 333 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._get_space` | 368 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._inference` | 391 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._text_generation` | 428 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._text_classification` | 454 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._token_classification` | 464 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._question_answering` | 474 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._summarization` | 492 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._translation` | 508 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._fill_mask` | 518 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._feature_extraction` | 528 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._image_classification` | 538 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._object_detection` | 570 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._image_to_text` | 580 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._text_to_image` | 590 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._text_to_speech` | 614 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._asr` | 638 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._zero_shot_classification` | 668 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._sentence_similarity` | 691 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._conversational` | 709 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool._whoami` | 735 |
| `argus/tools/integrations/media_ai/huggingface.py` | ⚙️ method | `HuggingFaceTool.get_schema` | 752 |
| `argus/tools/integrations/observability/arize.py` | 🏛️ class | `ArizeTool` | 19 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool.__init__` | 43 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_session` | 61 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._request` | 72 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool.execute` | 100 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._list_models` | 166 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_model` | 181 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_model_metrics` | 204 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_model_performance` | 237 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._log_prediction` | 275 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._log_actual` | 323 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._log_batch` | 362 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_drift_metrics` | 392 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_feature_drift` | 422 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_data_quality` | 451 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_missing_values` | 475 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._list_alerts` | 500 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._create_alert` | 519 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._delete_alert` | 556 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._list_monitors` | 577 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_monitor` | 596 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool._get_feature_importance` | 620 |
| `argus/tools/integrations/observability/arize.py` | ⚙️ method | `ArizeTool.get_schema` | 644 |
| `argus/tools/integrations/observability/mlflow_tool.py` | 🏛️ class | `MLflowTool` | 18 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool.__init__` | 41 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._get_session` | 56 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._request` | 67 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool.execute` | 97 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._list_experiments` | 174 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._get_experiment` | 195 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._create_experiment` | 222 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._delete_experiment` | 248 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._update_experiment` | 264 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._search_experiments` | 286 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._list_runs` | 317 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._get_run` | 346 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._create_run` | 361 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._update_run` | 394 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._delete_run` | 424 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._search_runs` | 440 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._log_metric` | 482 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._log_metrics` | 519 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._log_param` | 555 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._log_params` | 586 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._get_metric_history` | 614 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._set_tag` | 639 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._delete_tag` | 670 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._list_registered_models` | 697 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._get_registered_model` | 716 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._create_registered_model` | 735 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._delete_registered_model` | 761 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._update_registered_model` | 781 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._search_registered_models` | 803 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._list_model_versions` | 831 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._get_model_version` | 858 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._create_model_version` | 880 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._update_model_version` | 917 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._delete_model_version` | 946 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._transition_model_stage` | 970 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._list_artifacts` | 1003 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool._log_artifact` | 1025 |
| `argus/tools/integrations/observability/mlflow_tool.py` | ⚙️ method | `MLflowTool.get_schema` | 1069 |
| `argus/tools/integrations/observability/monocle.py` | 🏛️ class | `MonocleTool` | 19 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool.__init__` | 41 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_session` | 57 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._request` | 68 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool.execute` | 98 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._list_traces` | 161 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_trace` | 188 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._log_trace` | 203 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._list_spans` | 239 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_span` | 264 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._log_span` | 279 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._list_workflows` | 321 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_workflow` | 338 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._log_workflow` | 353 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._log_llm_call` | 391 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._log_retrieval` | 447 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._log_embedding` | 490 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_latency_stats` | 531 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_token_usage` | 558 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_cost_analysis` | 585 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._setup_tracing` | 613 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool._get_config` | 637 |
| `argus/tools/integrations/observability/monocle.py` | ⚙️ method | `MonocleTool.get_schema` | 648 |
| `argus/tools/integrations/observability/phoenix.py` | 🏛️ class | `PhoenixTool` | 19 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool.__init__` | 41 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_session` | 56 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._request` | 67 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool.execute` | 97 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._list_traces` | 163 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_trace` | 190 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._log_trace` | 205 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._delete_trace` | 240 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._list_spans` | 257 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_span` | 278 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._log_span` | 293 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._list_evaluations` | 341 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._log_evaluation` | 362 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_evaluation_metrics` | 399 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._list_datasets` | 420 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_dataset` | 437 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._create_dataset` | 452 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._list_projects` | 479 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_project` | 496 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_embedding_drift` | 512 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._get_clusters` | 532 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool._export_traces` | 557 |
| `argus/tools/integrations/observability/phoenix.py` | ⚙️ method | `PhoenixTool.get_schema` | 587 |
| `argus/tools/integrations/observability/wandb_weave.py` | 🏛️ class | `WandBWeaveTool` | 19 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool.__init__` | 42 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_session` | 61 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._request` | 74 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._graphql` | 105 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool.execute` | 130 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._list_calls` | 197 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_call` | 239 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._log_call` | 257 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._list_traces` | 318 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_trace` | 352 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._list_evaluations` | 371 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_evaluation` | 397 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._create_evaluation` | 415 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._log_evaluation_result` | 455 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._list_datasets` | 496 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_dataset` | 522 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._create_dataset` | 540 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._add_dataset_rows` | 572 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._list_models` | 599 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_model` | 625 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._log_model` | 643 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._list_ops` | 680 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_op` | 706 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_call_stats` | 727 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_cost_summary` | 759 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_latency_stats` | 790 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._list_projects` | 826 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool._get_project` | 863 |
| `argus/tools/integrations/observability/wandb_weave.py` | ⚙️ method | `WandBWeaveTool.get_schema` | 892 |
| `argus/tools/integrations/productivity/filesystem.py` | 🏛️ class | `FileSystemTool` | 19 |
| `argus/tools/integrations/productivity/filesystem.py` | ⚙️ method | `FileSystemTool.__init__` | 32 |
| `argus/tools/integrations/productivity/filesystem.py` | ⚙️ method | `FileSystemTool._safe_path` | 42 |
| `argus/tools/integrations/productivity/filesystem.py` | ⚙️ method | `FileSystemTool.execute` | 49 |
| `argus/tools/integrations/productivity/filesystem.py` | ⚙️ method | `FileSystemTool.get_schema` | 119 |
| `argus/tools/integrations/productivity/github.py` | 🏛️ class | `GitHubTool` | 18 |
| `argus/tools/integrations/productivity/github.py` | ⚙️ method | `GitHubTool.__init__` | 31 |
| `argus/tools/integrations/productivity/github.py` | ⚙️ method | `GitHubTool.execute` | 39 |
| `argus/tools/integrations/productivity/github.py` | ⚙️ method | `GitHubTool.get_schema` | 139 |
| `argus/tools/integrations/productivity/json_tool.py` | 🏛️ class | `JsonTool` | 18 |
| `argus/tools/integrations/productivity/json_tool.py` | ⚙️ method | `JsonTool.__init__` | 31 |
| `argus/tools/integrations/productivity/json_tool.py` | ⚙️ method | `JsonTool.execute` | 34 |
| `argus/tools/integrations/productivity/json_tool.py` | ⚙️ method | `JsonTool._get_path` | 97 |
| `argus/tools/integrations/productivity/json_tool.py` | ⚙️ method | `JsonTool.get_schema` | 108 |
| `argus/tools/integrations/productivity/python_repl.py` | 🏛️ class | `PythonReplTool` | 19 |
| `argus/tools/integrations/productivity/python_repl.py` | ⚙️ method | `PythonReplTool.__init__` | 32 |
| `argus/tools/integrations/productivity/python_repl.py` | ⚙️ method | `PythonReplTool.execute` | 43 |
| `argus/tools/integrations/productivity/python_repl.py` | ⚙️ method | `PythonReplTool.get_schema` | 102 |
| `argus/tools/integrations/productivity/shell.py` | 🏛️ class | `ShellTool` | 18 |
| `argus/tools/integrations/productivity/shell.py` | ⚙️ method | `ShellTool.__init__` | 34 |
| `argus/tools/integrations/productivity/shell.py` | ⚙️ method | `ShellTool._is_safe` | 46 |
| `argus/tools/integrations/productivity/shell.py` | ⚙️ method | `ShellTool.execute` | 53 |
| `argus/tools/integrations/productivity/shell.py` | ⚙️ method | `ShellTool.get_schema` | 105 |
| `argus/tools/integrations/productivity_tools/asana.py` | 🏛️ class | `AsanaTool` | 19 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool.__init__` | 45 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._get_session` | 63 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._request` | 75 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool.execute` | 99 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._list_workspaces` | 182 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._get_workspace` | 199 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._list_projects` | 209 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._get_project` | 239 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._create_project` | 248 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._update_project` | 275 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._delete_project` | 299 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._list_tasks` | 312 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._get_task` | 350 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._create_task` | 359 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._update_task` | 389 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._delete_task` | 422 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._complete_task` | 434 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._list_sections` | 451 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._create_section` | 481 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._add_task_to_section` | 504 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._list_tags` | 529 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._create_tag` | 559 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._add_tag_to_task` | 582 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool._search` | 608 |
| `argus/tools/integrations/productivity_tools/asana.py` | ⚙️ method | `AsanaTool.get_schema` | 641 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | 🏛️ class | `JiraTool` | 19 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool.__init__` | 42 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._get_session` | 62 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._request` | 78 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool.execute` | 110 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._list_projects` | 188 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._get_project` | 211 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._list_issues` | 225 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._get_issue` | 238 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._create_issue` | 266 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._update_issue` | 318 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._delete_issue` | 364 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._assign_issue` | 380 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._get_transitions` | 401 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._transition_issue` | 426 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._get_comments` | 451 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._add_comment` | 491 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._search` | 525 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._list_sprints` | 565 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool._add_to_sprint` | 601 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `JiraTool.get_schema` | 627 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | 🏛️ class | `ConfluenceTool` | 663 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool.__init__` | 685 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._get_session` | 702 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._request` | 718 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool.execute` | 747 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._list_spaces` | 802 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._get_space` | 825 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._list_pages` | 839 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._get_page` | 867 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._create_page` | 893 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._update_page` | 928 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._delete_page` | 968 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool._search` | 985 |
| `argus/tools/integrations/productivity_tools/atlassian.py` | ⚙️ method | `ConfluenceTool.get_schema` | 1025 |
| `argus/tools/integrations/productivity_tools/linear.py` | 🏛️ class | `LinearTool` | 18 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool.__init__` | 43 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._get_session` | 59 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._graphql` | 70 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool.execute` | 91 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._list_teams` | 174 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._get_team` | 206 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._list_issues` | 239 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._get_issue` | 308 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._create_issue` | 374 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._update_issue` | 437 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._delete_issue` | 493 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._list_projects` | 519 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._get_project` | 563 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._create_project` | 603 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._list_cycles` | 644 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._get_active_cycle` | 690 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._list_states` | 722 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._list_labels` | 765 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._create_label` | 806 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._search` | 851 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._list_users` | 899 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool._me` | 931 |
| `argus/tools/integrations/productivity_tools/linear.py` | ⚙️ method | `LinearTool.get_schema` | 953 |
| `argus/tools/integrations/productivity_tools/notion.py` | 🏛️ class | `NotionTool` | 18 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool.__init__` | 43 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._get_session` | 59 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._request` | 71 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool.execute` | 98 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._extract_title` | 170 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._get_page` | 193 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._create_page` | 215 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._update_page` | 270 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._archive_page` | 291 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._get_database` | 308 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._query_database` | 336 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._create_database` | 374 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._get_blocks` | 404 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._append_blocks` | 446 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._delete_block` | 486 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._search` | 503 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._list_users` | 537 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._get_user` | 560 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool._me` | 577 |
| `argus/tools/integrations/productivity_tools/notion.py` | ⚙️ method | `NotionTool.get_schema` | 589 |
| `argus/tools/integrations/search/arxiv_tool.py` | 🏛️ class | `ArxivTool` | 17 |
| `argus/tools/integrations/search/arxiv_tool.py` | ⚙️ method | `ArxivTool.__init__` | 30 |
| `argus/tools/integrations/search/arxiv_tool.py` | ⚙️ method | `ArxivTool.execute` | 40 |
| `argus/tools/integrations/search/arxiv_tool.py` | ⚙️ method | `ArxivTool.get_schema` | 98 |
| `argus/tools/integrations/search/brave.py` | 🏛️ class | `BraveTool` | 18 |
| `argus/tools/integrations/search/brave.py` | ⚙️ method | `BraveTool.__init__` | 33 |
| `argus/tools/integrations/search/brave.py` | ⚙️ method | `BraveTool.execute` | 41 |
| `argus/tools/integrations/search/brave.py` | ⚙️ method | `BraveTool.get_schema` | 101 |
| `argus/tools/integrations/search/duckduckgo.py` | 🏛️ class | `DuckDuckGoTool` | 17 |
| `argus/tools/integrations/search/duckduckgo.py` | ⚙️ method | `DuckDuckGoTool.__init__` | 31 |
| `argus/tools/integrations/search/duckduckgo.py` | ⚙️ method | `DuckDuckGoTool.execute` | 43 |
| `argus/tools/integrations/search/duckduckgo.py` | ⚙️ method | `DuckDuckGoTool.get_schema` | 99 |
| `argus/tools/integrations/search/exa.py` | 🏛️ class | `ExaTool` | 18 |
| `argus/tools/integrations/search/exa.py` | ⚙️ method | `ExaTool.__init__` | 31 |
| `argus/tools/integrations/search/exa.py` | ⚙️ method | `ExaTool.execute` | 39 |
| `argus/tools/integrations/search/exa.py` | ⚙️ method | `ExaTool.get_schema` | 110 |
| `argus/tools/integrations/search/tavily.py` | 🏛️ class | `TavilyTool` | 18 |
| `argus/tools/integrations/search/tavily.py` | ⚙️ method | `TavilyTool.__init__` | 31 |
| `argus/tools/integrations/search/tavily.py` | ⚙️ method | `TavilyTool.execute` | 41 |
| `argus/tools/integrations/search/tavily.py` | ⚙️ method | `TavilyTool.get_schema` | 108 |
| `argus/tools/integrations/search/wikipedia.py` | 🏛️ class | `WikipediaTool` | 17 |
| `argus/tools/integrations/search/wikipedia.py` | ⚙️ method | `WikipediaTool.__init__` | 30 |
| `argus/tools/integrations/search/wikipedia.py` | ⚙️ method | `WikipediaTool.execute` | 40 |
| `argus/tools/integrations/search/wikipedia.py` | ⚙️ method | `WikipediaTool.get_schema` | 113 |
| `argus/tools/integrations/vectordb/chroma.py` | 🏛️ class | `ChromaTool` | 20 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool.__init__` | 42 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._get_client` | 62 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._get_embedding_function` | 83 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool.execute` | 105 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._get_collection_obj` | 162 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._add` | 172 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._query` | 209 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._get` | 264 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._update` | 307 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._delete` | 339 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._create_collection` | 368 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._delete_collection` | 392 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._list_collections` | 409 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._get_collection` | 422 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._count` | 439 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool._peek` | 455 |
| `argus/tools/integrations/vectordb/chroma.py` | ⚙️ method | `ChromaTool.get_schema` | 483 |
| `argus/tools/integrations/vectordb/mongodb.py` | 🏛️ class | `MongoDBTool` | 19 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool.__init__` | 41 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._get_client` | 59 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool.execute` | 69 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._insert` | 141 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._insert_one` | 172 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._find` | 200 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._find_one` | 241 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._update` | 269 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._update_one` | 306 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._delete` | 341 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._delete_one` | 364 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._count` | 387 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._aggregate` | 410 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._vector_search` | 441 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._create_index` | 497 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._create_vector_index` | 523 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._list_indexes` | 563 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._drop_index` | 586 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._list_collections` | 612 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._create_collection` | 629 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._drop_collection` | 650 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool._list_databases` | 671 |
| `argus/tools/integrations/vectordb/mongodb.py` | ⚙️ method | `MongoDBTool.get_schema` | 682 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | 🏛️ class | `PineconeTool` | 19 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool.__init__` | 41 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._get_client` | 59 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._get_index` | 69 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool.execute` | 78 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._upsert` | 134 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._query` | 172 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._fetch` | 219 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._delete` | 246 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._update` | 276 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._list_indexes` | 304 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._create_index` | 323 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._delete_index` | 355 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._describe_index` | 372 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool._describe_index_stats` | 393 |
| `argus/tools/integrations/vectordb/pinecone_tool.py` | ⚙️ method | `PineconeTool.get_schema` | 409 |
| `argus/tools/integrations/vectordb/qdrant.py` | 🏛️ class | `QdrantTool` | 19 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool.__init__` | 41 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._get_client` | 60 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool.execute` | 74 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._upsert` | 128 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._search` | 170 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._retrieve` | 224 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._delete` | 270 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._scroll` | 314 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._create_collection` | 353 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._delete_collection` | 389 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._list_collections` | 406 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._get_collection` | 419 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool._count` | 438 |
| `argus/tools/integrations/vectordb/qdrant.py` | ⚙️ method | `QdrantTool.get_schema` | 455 |
| `argus/tools/integrations/web/jina_reader.py` | 🏛️ class | `JinaReaderTool` | 17 |
| `argus/tools/integrations/web/jina_reader.py` | ⚙️ method | `JinaReaderTool.__init__` | 32 |
| `argus/tools/integrations/web/jina_reader.py` | ⚙️ method | `JinaReaderTool.execute` | 41 |
| `argus/tools/integrations/web/jina_reader.py` | ⚙️ method | `JinaReaderTool.get_schema` | 87 |
| `argus/tools/integrations/web/requests_tool.py` | 🏛️ class | `RequestsTool` | 17 |
| `argus/tools/integrations/web/requests_tool.py` | ⚙️ method | `RequestsTool.__init__` | 30 |
| `argus/tools/integrations/web/requests_tool.py` | ⚙️ method | `RequestsTool.execute` | 40 |
| `argus/tools/integrations/web/requests_tool.py` | ⚙️ method | `RequestsTool.get_schema` | 109 |
| `argus/tools/integrations/web/scraper.py` | 🏛️ class | `WebScraperTool` | 17 |
| `argus/tools/integrations/web/scraper.py` | ⚙️ method | `WebScraperTool.__init__` | 30 |
| `argus/tools/integrations/web/scraper.py` | ⚙️ method | `WebScraperTool.execute` | 38 |
| `argus/tools/integrations/web/scraper.py` | ⚙️ method | `WebScraperTool.get_schema` | 115 |
| `argus/tools/integrations/web/youtube.py` | 🏛️ class | `YouTubeTool` | 18 |
| `argus/tools/integrations/web/youtube.py` | ⚙️ method | `YouTubeTool.__init__` | 31 |
| `argus/tools/integrations/web/youtube.py` | ⚙️ method | `YouTubeTool._extract_video_id` | 34 |
| `argus/tools/integrations/web/youtube.py` | ⚙️ method | `YouTubeTool.execute` | 46 |
| `argus/tools/integrations/web/youtube.py` | ⚙️ method | `YouTubeTool.get_schema` | 111 |
| `argus/tools/registry.py` | 🏛️ class | `ToolRegistry` | 25 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.__init__` | 45 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.register` | 52 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.register_class` | 68 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.unregister` | 86 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.get` | 102 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.get_or_raise` | 114 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.has` | 131 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.list_all` | 143 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.list_by_category` | 152 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.get_schemas` | 167 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.get_stats` | 176 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.clear` | 191 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.__len__` | 197 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.__contains__` | 202 |
| `argus/tools/registry.py` | ⚙️ method | `ToolRegistry.__iter__` | 206 |
| `argus/tools/registry.py` | 🔧 function | `get_default_registry` | 220 |
| `argus/tools/registry.py` | 🔧 function | `register_tool` | 255 |
| `argus/tools/registry.py` | 🔧 function | `get_tool` | 266 |
| `argus/tools/registry.py` | 🔧 function | `list_tools` | 278 |
