"""Configuration management for Argus Terminal Sandbox."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json


@dataclass
class ThemeConfig:
    """Theme configuration."""
    
    name: Literal["amber", "green"] = "amber"
    scanlines: bool = True
    glow_effect: bool = True


@dataclass
class DebateConfig:
    """Debate configuration."""
    
    max_rounds: int = 5
    default_prior: float = 0.5
    convergence_threshold: float = 0.01
    specialists_count: int = 3


@dataclass
class ProviderConfig:
    """LLM Provider configuration."""
    
    default_provider: str = "gemini"
    default_model: str = "gemini-1.5-flash"
    api_keys: dict = field(default_factory=dict)


@dataclass
class AppConfig:
    """Main application configuration."""
    
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    show_help_on_start: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        """Load configuration from file."""
        if path is None:
            path = Path.home() / ".argus_terminal" / "config.json"
        
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return cls(
                    theme=ThemeConfig(**data.get("theme", {})),
                    debate=DebateConfig(**data.get("debate", {})),
                    provider=ProviderConfig(**data.get("provider", {})),
                    show_help_on_start=data.get("show_help_on_start", True),
                    log_level=data.get("log_level", "INFO"),
                )
            except Exception:
                pass
        
        return cls()
    
    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = Path.home() / ".argus_terminal" / "config.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "theme": {
                "name": self.theme.name,
                "scanlines": self.theme.scanlines,
                "glow_effect": self.theme.glow_effect,
            },
            "debate": {
                "max_rounds": self.debate.max_rounds,
                "default_prior": self.debate.default_prior,
                "convergence_threshold": self.debate.convergence_threshold,
                "specialists_count": self.debate.specialists_count,
            },
            "provider": {
                "default_provider": self.provider.default_provider,
                "default_model": self.provider.default_model,
                "api_keys": self.provider.api_keys,
            },
            "show_help_on_start": self.show_help_on_start,
            "log_level": self.log_level,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
