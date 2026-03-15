"""
Evidence Half-Life Registry for CHRONOS.

Provides configurable per-type decay parameters. Users can override
default half-lives for each evidence category or register custom ones.

Default half-lives:
    EMPIRICAL (RCT data)         : 5 years
    MARKET signals               : 24 hours
    EXPERT opinion               : 2 years
    STATISTICAL analysis         : 3 years
    LITERATURE (published)       : 4 years
    COMPUTATIONAL (simulations)  : 3 years
    EMERGENT (live-stream)       : 12 hours
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from argus.chronos.temporal_cdag import (
    EvidenceCategory,
    DecayFunction,
    DEFAULT_HALF_LIVES,
)

logger = logging.getLogger(__name__)


@dataclass
class HalfLifeConfig:
    """
    Configuration for a single evidence category's half-life.

    Attributes:
        category: Evidence category
        half_life_hours: Half-life in hours
        description: Human-readable description
    """
    category: EvidenceCategory
    half_life_hours: float
    description: str = ""

    @classmethod
    def from_years(
        cls,
        category: EvidenceCategory,
        years: float,
        description: str = "",
    ) -> "HalfLifeConfig":
        """Create config from years."""
        return cls(
            category=category,
            half_life_hours=years * 365.25 * 24.0,
            description=description or f"{category.value}: {years}y half-life",
        )

    @classmethod
    def from_days(
        cls,
        category: EvidenceCategory,
        days: float,
        description: str = "",
    ) -> "HalfLifeConfig":
        """Create config from days."""
        return cls(
            category=category,
            half_life_hours=days * 24.0,
            description=description or f"{category.value}: {days}d half-life",
        )

    @property
    def half_life_years(self) -> float:
        """Half-life in years."""
        return self.half_life_hours / (365.25 * 24.0)

    @property
    def half_life_days(self) -> float:
        """Half-life in days."""
        return self.half_life_hours / 24.0


class EvidenceHalfLifeRegistry:
    """
    Registry of evidence half-life configurations.

    Manages decay parameters for all evidence categories. Users can
    override defaults or add custom categories.

    Example:
        >>> registry = EvidenceHalfLifeRegistry(
        ...     empirical_years=5.0,
        ...     market_hours=24.0,
        ...     expert_years=2.0,
        ...     statistical_years=3.0,
        ... )
        >>> decay_fn = registry.get_decay_function(EvidenceCategory.EMPIRICAL)
        >>> weight = decay_fn.compute_weight(0.9, age_hours=365.25*24*3)
    """

    def __init__(
        self,
        empirical_years: float = 5.0,
        market_hours: float = 24.0,
        expert_years: float = 2.0,
        statistical_years: float = 3.0,
        literature_years: float = 4.0,
        computational_years: float = 3.0,
        emergent_hours: float = 12.0,
    ):
        self._configs: dict[EvidenceCategory, HalfLifeConfig] = {}

        # Register defaults
        self.register(HalfLifeConfig.from_years(
            EvidenceCategory.EMPIRICAL, empirical_years,
        ))
        self.register(HalfLifeConfig(
            EvidenceCategory.MARKET, market_hours,
            f"market: {market_hours}h half-life",
        ))
        self.register(HalfLifeConfig.from_years(
            EvidenceCategory.EXPERT, expert_years,
        ))
        self.register(HalfLifeConfig.from_years(
            EvidenceCategory.STATISTICAL, statistical_years,
        ))
        self.register(HalfLifeConfig.from_years(
            EvidenceCategory.LITERATURE, literature_years,
        ))
        self.register(HalfLifeConfig.from_years(
            EvidenceCategory.COMPUTATIONAL, computational_years,
        ))
        self.register(HalfLifeConfig(
            EvidenceCategory.EMERGENT, emergent_hours,
            f"emergent: {emergent_hours}h half-life",
        ))

        logger.info(
            f"EvidenceHalfLifeRegistry initialized with "
            f"{len(self._configs)} categories"
        )

    def register(self, config: HalfLifeConfig) -> None:
        """Register or override a half-life configuration."""
        self._configs[config.category] = config
        logger.debug(
            f"Registered half-life: {config.category.value} = "
            f"{config.half_life_hours:.1f}h"
        )

    def get_config(self, category: EvidenceCategory) -> HalfLifeConfig:
        """
        Get half-life config for a category.

        Args:
            category: Evidence category

        Returns:
            HalfLifeConfig for the category (or default)
        """
        if category in self._configs:
            return self._configs[category]

        # Return default
        default_hours = DEFAULT_HALF_LIVES.get(
            category, 3.0 * 365.25 * 24.0,
        )
        return HalfLifeConfig(
            category=category,
            half_life_hours=default_hours,
            description=f"default: {category.value}",
        )

    def get_decay_function(self, category: EvidenceCategory) -> DecayFunction:
        """
        Get decay function for a category.

        Args:
            category: Evidence category

        Returns:
            DecayFunction with appropriate half-life
        """
        config = self.get_config(category)
        return DecayFunction(
            category=category,
            half_life_hours=config.half_life_hours,
        )

    @property
    def all_configs(self) -> list[HalfLifeConfig]:
        """Return all registered configurations."""
        return list(self._configs.values())

    def summary(self) -> dict[str, float]:
        """Return summary of all half-lives in years."""
        return {
            config.category.value: config.half_life_years
            for config in self._configs.values()
        }

    def __repr__(self) -> str:
        parts = []
        for config in self._configs.values():
            if config.half_life_hours >= 365.25 * 24:
                parts.append(
                    f"{config.category.value}={config.half_life_years:.1f}y"
                )
            else:
                parts.append(
                    f"{config.category.value}={config.half_life_hours:.0f}h"
                )
        return f"EvidenceHalfLifeRegistry({', '.join(parts)})"
