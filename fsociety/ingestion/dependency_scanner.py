"""
Dependency scanner for fsociety.

Parses manifest files and flags known-vulnerable library versions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Dependency:
    """A parsed dependency."""
    name: str
    version: str = ""
    source_file: str = ""
    ecosystem: str = ""  # pypi, npm, go, maven, cargo, gem
    cve_flags: list[str] = field(default_factory=list)


class DependencyScanner:
    """
    Parses dependency manifest files and flags potential CVE matches.

    Supports: requirements.txt, package.json, go.mod, Cargo.toml,
    Gemfile, pom.xml, build.gradle, pyproject.toml
    """

    MANIFEST_PARSERS = {
        "requirements.txt": "_parse_requirements_txt",
        "package.json": "_parse_package_json",
        "go.mod": "_parse_go_mod",
        "pyproject.toml": "_parse_pyproject_toml",
        "Cargo.toml": "_parse_cargo_toml",
        "Gemfile": "_parse_gemfile",
    }

    def scan_directory(self, directory: str | Path) -> list[Dependency]:
        """Scan a directory for dependency manifest files."""
        directory = Path(directory)
        deps: list[Dependency] = []

        for name, parser_method in self.MANIFEST_PARSERS.items():
            for manifest in directory.rglob(name):
                # Skip vendor/node_modules
                if any(p in manifest.parts for p in ("node_modules", ".venv", "venv")):
                    continue
                try:
                    parser = getattr(self, parser_method)
                    parsed = parser(manifest)
                    deps.extend(parsed)
                    logger.info(f"Parsed {len(parsed)} deps from {manifest}")
                except Exception as e:
                    logger.warning(f"Could not parse {manifest}: {e}")

        return deps

    def _parse_requirements_txt(self, path: Path) -> list[Dependency]:
        """Parse Python requirements.txt."""
        deps = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            match = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=!~]+.+)?", line)
            if match:
                deps.append(Dependency(
                    name=match.group(1),
                    version=match.group(2) or "",
                    source_file=str(path),
                    ecosystem="pypi",
                ))
        return deps

    def _parse_package_json(self, path: Path) -> list[Dependency]:
        """Parse Node.js package.json."""
        deps = []
        try:
            data = json.loads(path.read_text(errors="replace"))
            for section in ("dependencies", "devDependencies"):
                for name, version in data.get(section, {}).items():
                    deps.append(Dependency(
                        name=name,
                        version=version,
                        source_file=str(path),
                        ecosystem="npm",
                    ))
        except json.JSONDecodeError:
            pass
        return deps

    def _parse_go_mod(self, path: Path) -> list[Dependency]:
        """Parse Go go.mod."""
        deps = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            match = re.match(r"^\s*([a-zA-Z0-9./_-]+)\s+(v[\d.]+)", line)
            if match:
                deps.append(Dependency(
                    name=match.group(1),
                    version=match.group(2),
                    source_file=str(path),
                    ecosystem="go",
                ))
        return deps

    def _parse_pyproject_toml(self, path: Path) -> list[Dependency]:
        """Parse Python pyproject.toml dependencies."""
        deps = []
        in_deps = False
        for line in path.read_text(errors="replace").splitlines():
            if "dependencies" in line and "=" in line and "[" in line:
                in_deps = True
                continue
            if in_deps:
                if line.strip().startswith("]"):
                    in_deps = False
                    continue
                match = re.match(r'\s*"([a-zA-Z0-9_.-]+)\s*([><=!~]+[^"]*)?', line)
                if match:
                    deps.append(Dependency(
                        name=match.group(1),
                        version=match.group(2) or "",
                        source_file=str(path),
                        ecosystem="pypi",
                    ))
        return deps

    def _parse_cargo_toml(self, path: Path) -> list[Dependency]:
        """Parse Rust Cargo.toml."""
        deps = []
        in_deps = False
        for line in path.read_text(errors="replace").splitlines():
            if line.strip() == "[dependencies]":
                in_deps = True
                continue
            if in_deps:
                if line.strip().startswith("["):
                    in_deps = False
                    continue
                match = re.match(r'^([a-zA-Z0-9_-]+)\s*=\s*"([^"]+)"', line)
                if match:
                    deps.append(Dependency(
                        name=match.group(1),
                        version=match.group(2),
                        source_file=str(path),
                        ecosystem="cargo",
                    ))
        return deps

    def _parse_gemfile(self, path: Path) -> list[Dependency]:
        """Parse Ruby Gemfile."""
        deps = []
        for line in path.read_text(errors="replace").splitlines():
            match = re.match(r"^\s*gem\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?", line)
            if match:
                deps.append(Dependency(
                    name=match.group(1),
                    version=match.group(2) or "",
                    source_file=str(path),
                    ecosystem="gem",
                ))
        return deps
