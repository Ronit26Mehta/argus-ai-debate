"""Screens package for Argus Terminal."""

from argus_terminal.screens.main import MainScreen
from argus_terminal.screens.debate import DebateScreen
from argus_terminal.screens.providers import ProvidersScreen
from argus_terminal.screens.tools import ToolsScreen
from argus_terminal.screens.knowledge import KnowledgeScreen
from argus_terminal.screens.benchmark import BenchmarkScreen
from argus_terminal.screens.settings import SettingsScreen
from argus_terminal.screens.help import HelpScreen

__all__ = [
    "MainScreen",
    "DebateScreen",
    "ProvidersScreen",
    "ToolsScreen",
    "KnowledgeScreen",
    "BenchmarkScreen",
    "SettingsScreen",
    "HelpScreen",
]
