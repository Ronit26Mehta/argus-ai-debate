"""Local JSON folder storage for ARGUS Sandbox runs."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (slug or "sandbox-run")[:max_len]


@dataclass
class SandboxRunPaths:
    """Filesystem paths for a single sandbox run."""

    run_id: str
    root: Path
    stages_dir: Path
    events_file: Path
    summary_file: Path


class JsonFolderStore:
    """Persist run artifacts as JSON files with append-only event logs."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create_run(self, proposition: str) -> SandboxRunPaths:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{ts}_{_slugify(proposition)}"
        root = self.base_dir / run_id
        stages_dir = root / "stages"
        root.mkdir(parents=True, exist_ok=True)
        stages_dir.mkdir(parents=True, exist_ok=True)

        events_file = root / "events.jsonl"
        summary_file = root / "summary.json"
        events_file.touch(exist_ok=True)

        return SandboxRunPaths(
            run_id=run_id,
            root=root,
            stages_dir=stages_dir,
            events_file=events_file,
            summary_file=summary_file,
        )

    def append_event(self, paths: SandboxRunPaths, event: dict[str, Any]) -> None:
        """Append one event as JSON line to avoid loading full history in memory."""
        payload = dict(event)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self._lock:
            with paths.events_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def write_stage(self, paths: SandboxRunPaths, stage: str, data: dict[str, Any]) -> Path:
        safe = _slugify(stage)
        out = paths.stages_dir / f"{safe}.json"
        with self._lock:
            with out.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=True)
        return out

    def write_summary(self, paths: SandboxRunPaths, summary: dict[str, Any]) -> Path:
        with self._lock:
            with paths.summary_file.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=True)
        return paths.summary_file
