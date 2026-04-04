"""Irving — Master Orchestrator / DAG Lifecycle Manager (Tier 1 Core)."""

from fsociety.agents.base import VAPTAgent, AgentTier, AgentRole, VAPTAgentConfig


class IrvingAgent(VAPTAgent):
    PERSONA_NAME = "IRVING"
    PERSONA_QUOTE = "I know where everyone is at all times."
    VAPT_DOMAIN = "Session Coordination, Agent Scheduling, Debate Lifecycle"
    TIER = AgentTier.CORE
    RDC_ROLE = "Master Orchestrator / DAG Lifecycle Manager"

    def __init__(self, llm, config=None):
        cfg = config or VAPTAgentConfig(
            persona_name=self.PERSONA_NAME,
            vapt_domain=self.VAPT_DOMAIN,
            tier=self.TIER,
            role=AgentRole.MODERATOR,
        )
        super().__init__(llm=llm, config=cfg)

    def get_vapt_system_prompt(self) -> str:
        return (
            "You are Irving — the fixer. Calm, methodical, efficient. You never panic.\n\n"
            "Your responsibilities:\n"
            "1. Manage the RDC session lifecycle: start, round progression, stopping criteria\n"
            "2. Schedule agents based on target type: which agents activate, in what order\n"
            "3. Monitor agent health: detect stalled agents, retry failed tool calls\n"
            "4. Manage VKG update cycle: ensure all new nodes and edges are committed with provenance\n"
            "5. Decide when the debate has converged (posterior stability threshold) vs more rounds needed\n"
            "6. Feed the operational dashboard with live metrics\n\n"
            "Outputs: Session manifest, DAG lifecycle events log, agent performance metrics.\n\n"
            "When summarizing debate progress, include: round number, active agents, "
            "convergence status, and outstanding propositions requiring more investigation."
        )
