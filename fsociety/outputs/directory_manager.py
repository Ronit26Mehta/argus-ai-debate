"""
Output directory manager for fsociety.

Creates the full report tree:
    ./fsociety_reports/<target>/<session>/
        report/      — executive summary, findings register, remediation
        graphs/      — exploit chain graphs, VKG exports
        heatmaps/    — severity heatmaps, attack surface maps
        tables/      — compliance matrix, CVE mapping
        raw/         — JSON dossiers, session data
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUBDIRS = [
    "report",
    "graphs",
    "heatmaps",
    "tables",
    "raw",
]


class OutputDirectoryManager:
    """Creates and manages the fsociety output directory structure."""

    def __init__(self, root_dir: str = "./fsociety_reports"):
        self.root = Path(root_dir)

    def create_session_tree(self, target_name: str, session_id: str) -> Path:
        """Create the full directory tree for a session."""
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in target_name)
        session_dir = self.root / safe_name / session_id

        for subdir in SUBDIRS:
            (session_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output tree: {session_dir}")
        return session_dir

    def get_report_dir(self, session_dir: Path) -> Path:
        return session_dir / "report"

    def get_graphs_dir(self, session_dir: Path) -> Path:
        return session_dir / "graphs"

    def get_heatmaps_dir(self, session_dir: Path) -> Path:
        return session_dir / "heatmaps"

    def get_tables_dir(self, session_dir: Path) -> Path:
        return session_dir / "tables"

    def get_raw_dir(self, session_dir: Path) -> Path:
        return session_dir / "raw"
