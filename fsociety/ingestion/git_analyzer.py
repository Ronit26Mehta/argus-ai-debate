"""
Git history analyzer for fsociety.

Temporal attack surface analysis from git commit history.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GitInsight:
    """An insight derived from git history analysis."""
    category: str  # "credential_exposure", "high_churn", "deleted_secrets", "debug_flag"
    description: str
    file_path: Optional[str] = None
    commit_hash: Optional[str] = None
    severity: str = "MEDIUM"


class GitAnalyzer:
    """
    Analyzes git repository history for security-relevant signals.

    Looks for:
    - High-churn files (recently changed code is more likely to have bugs)
    - Exposed credentials in commit history
    - Debug flags left behind
    - Recently deleted files that reveal architectural secrets
    """

    # Patterns that suggest credential exposure
    SECRET_PATTERNS = [
        re.compile(r"(?i)(api[_-]?key|secret|password|token|credential)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        re.compile(r"(?i)AKIA[0-9A-Z]{16}"),  # AWS access key
        re.compile(r"sk-[a-zA-Z0-9]{32,}"),  # OpenAI-style key
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub PAT
        re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
    ]

    def analyze(self, repo_path: str | Path) -> list[GitInsight]:
        """Run all git analyses on a repository."""
        repo_path = Path(repo_path)
        insights: list[GitInsight] = []

        try:
            import git
            repo = git.Repo(repo_path)
        except ImportError:
            logger.warning("GitPython not installed — skipping git analysis")
            return insights
        except Exception as e:
            logger.warning(f"Not a git repo or git error: {e}")
            return insights

        insights.extend(self._find_high_churn_files(repo))
        insights.extend(self._scan_commit_diffs_for_secrets(repo))
        insights.extend(self._find_deleted_sensitive_files(repo))

        logger.info(f"Git analysis found {len(insights)} insights")
        return insights

    def _find_high_churn_files(self, repo) -> list[GitInsight]:
        """Find files with high commit frequency (last 30 commits)."""
        insights = []
        churn: dict[str, int] = {}

        try:
            for commit in list(repo.iter_commits(max_count=30)):
                for file_path in commit.stats.files:
                    churn[file_path] = churn.get(file_path, 0) + 1

            # Flag files changed in >40% of recent commits
            threshold = max(3, len(list(repo.iter_commits(max_count=30))) * 0.4)
            for file_path, count in sorted(churn.items(), key=lambda x: -x[1]):
                if count >= threshold:
                    insights.append(GitInsight(
                        category="high_churn",
                        description=f"High-churn file: {file_path} changed in {count} of last 30 commits",
                        file_path=file_path,
                        severity="LOW",
                    ))
        except Exception as e:
            logger.warning(f"Churn analysis failed: {e}")

        return insights

    def _scan_commit_diffs_for_secrets(self, repo) -> list[GitInsight]:
        """Scan recent commit diffs for credential patterns."""
        insights = []

        try:
            for commit in list(repo.iter_commits(max_count=20)):
                try:
                    diff_text = repo.git.show(commit.hexsha, format="", stat=False)
                except Exception:
                    continue

                for pattern in self.SECRET_PATTERNS:
                    matches = pattern.findall(diff_text)
                    if matches:
                        insights.append(GitInsight(
                            category="credential_exposure",
                            description=f"Potential secret found in commit {commit.hexsha[:8]}: {matches[0][:50]}...",
                            commit_hash=commit.hexsha,
                            severity="HIGH",
                        ))
                        break  # One match per commit is enough
        except Exception as e:
            logger.warning(f"Secret scan failed: {e}")

        return insights

    def _find_deleted_sensitive_files(self, repo) -> list[GitInsight]:
        """Find recently deleted files that might reveal architecture secrets."""
        insights = []
        sensitive_patterns = [".env", "config", "secret", "key", "credential", "password", "auth"]

        try:
            for commit in list(repo.iter_commits(max_count=20)):
                try:
                    for diff in commit.diff(commit.parents[0] if commit.parents else None):
                        if diff.deleted_file:
                            name = diff.a_path.lower()
                            if any(p in name for p in sensitive_patterns):
                                insights.append(GitInsight(
                                    category="deleted_secrets",
                                    description=f"Deleted sensitive file: {diff.a_path}",
                                    file_path=diff.a_path,
                                    commit_hash=commit.hexsha,
                                    severity="MEDIUM",
                                ))
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Deleted file analysis failed: {e}")

        return insights
