"""
fsociety agents — the full Mr. Robot cast.

13 agents organized into three tiers:
  Tier 1 (Core):       Elliot, MrRobot, Darlene, Whiterose, Irving
  Tier 2 (Specialist): Romero, Mobley, Trenton, Tyrell, Angela, Dom
  Tier 3 (Output):     Leon, Cisco
"""

from fsociety.agents.base import VAPTAgent, AgentTier
from fsociety.agents.elliot import ElliotAgent
from fsociety.agents.mrrobot import MrRobotAgent
from fsociety.agents.darlene import DarleneAgent
from fsociety.agents.whiterose import WhiteroseAgent
from fsociety.agents.irving import IrvingAgent
from fsociety.agents.romero import RomeroAgent
from fsociety.agents.mobley import MobleyAgent
from fsociety.agents.trenton import TrentonAgent
from fsociety.agents.tyrell import TyrellAgent
from fsociety.agents.angela import AngelaAgent
from fsociety.agents.dom import DomAgent
from fsociety.agents.leon import LeonAgent
from fsociety.agents.cisco import CiscoAgent

ALL_AGENTS = [
    ElliotAgent, MrRobotAgent, DarleneAgent, WhiteroseAgent, IrvingAgent,
    RomeroAgent, MobleyAgent, TrentonAgent, TyrellAgent, AngelaAgent, DomAgent,
    LeonAgent, CiscoAgent,
]

TIER1_CORE = [ElliotAgent, MrRobotAgent, DarleneAgent, WhiteroseAgent, IrvingAgent]
TIER2_SPECIALIST = [RomeroAgent, MobleyAgent, TrentonAgent, TyrellAgent, AngelaAgent, DomAgent]
TIER3_OUTPUT = [LeonAgent, CiscoAgent]

__all__ = [
    "VAPTAgent", "AgentTier",
    "ElliotAgent", "MrRobotAgent", "DarleneAgent", "WhiteroseAgent", "IrvingAgent",
    "RomeroAgent", "MobleyAgent", "TrentonAgent", "TyrellAgent", "AngelaAgent", "DomAgent",
    "LeonAgent", "CiscoAgent",
    "ALL_AGENTS", "TIER1_CORE", "TIER2_SPECIALIST", "TIER3_OUTPUT",
]
