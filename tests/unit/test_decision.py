"""
Unit tests for decision layer.
"""

import pytest
import math


class TestBayesianUpdater:
    """Tests for Bayesian updating."""
    
    def test_log_odds_conversion(self):
        """Test log-odds conversion functions."""
        from argus.decision.bayesian import log_odds, from_log_odds
        
        # Test round-trip
        p = 0.7
        lo = log_odds(p)
        p_back = from_log_odds(lo)
        
        assert abs(p - p_back) < 0.001
    
    def test_bayesian_update(self):
        """Test Bayesian update."""
        from argus.decision.bayesian import BayesianUpdater
        
        updater = BayesianUpdater()
        
        # With positive evidence
        posterior = updater.update(
            prior=0.5,
            evidence_contributions=[0.5, 0.3],
        )
        
        assert posterior > 0.5  # Should increase
    
    def test_bayesian_update_with_negative(self):
        """Test Bayesian update with negative evidence."""
        from argus.decision.bayesian import BayesianUpdater
        
        updater = BayesianUpdater()
        
        posterior = updater.update(
            prior=0.5,
            evidence_contributions=[-0.5, -0.3],
        )
        
        assert posterior < 0.5  # Should decrease
    
    def test_temperature_scaling(self):
        """Test temperature effect on updates."""
        from argus.decision.bayesian import BayesianUpdater
        
        updater_low_temp = BayesianUpdater(temperature=0.5)
        updater_high_temp = BayesianUpdater(temperature=2.0)
        
        contributions = [0.5]
        
        post_low = updater_low_temp.update(0.5, contributions)
        post_high = updater_high_temp.update(0.5, contributions)
        
        # Lower temperature = more extreme
        assert abs(post_low - 0.5) > abs(post_high - 0.5)


class TestEIG:
    """Tests for Expected Information Gain."""
    
    def test_entropy(self):
        """Test entropy computation."""
        from argus.decision.eig import entropy
        
        # Max entropy at 0.5
        h_max = entropy(0.5)
        h_low = entropy(0.1)
        h_high = entropy(0.9)
        
        assert h_max > h_low
        assert h_max > h_high
    
    def test_eig_estimate(self):
        """Test EIG estimation."""
        from argus.decision.eig import estimate_eig
        
        # Higher EIG near 0.5 (max uncertainty)
        eig_uncertain = estimate_eig(prior=0.5)
        eig_certain = estimate_eig(prior=0.95)
        
        assert eig_uncertain > eig_certain
    
    def test_action_ranking(self):
        """Test action ranking by EIG."""
        from argus.decision.eig import ActionCandidate, rank_actions_by_eig
        
        actions = [
            ActionCandidate(id="a1", name="Action 1", cost=1.0, target_props=["p1"]),
            ActionCandidate(id="a2", name="Action 2", cost=2.0, target_props=["p1"]),
        ]
        
        priors = {"p1": 0.5}
        
        ranked = rank_actions_by_eig(actions, priors)
        
        assert len(ranked) == 2
        # Lower cost should rank higher (same EIG)
        assert ranked[0].id == "a1"


class TestCalibration:
    """Tests for calibration metrics."""
    
    def test_brier_score(self):
        """Test Brier score computation."""
        from argus.decision.calibration import compute_brier_score
        
        # Perfect predictions
        confidences = [1.0, 0.0, 1.0]
        outcomes = [1, 0, 1]
        
        brier = compute_brier_score(confidences, outcomes)
        
        assert brier == 0.0
    
    def test_brier_score_worst(self):
        """Test Brier score for worst predictions."""
        from argus.decision.calibration import compute_brier_score
        
        # Completely wrong
        confidences = [1.0, 1.0]
        outcomes = [0, 0]
        
        brier = compute_brier_score(confidences, outcomes)
        
        assert brier == 1.0
    
    def test_compute_ece(self):
        """Test ECE computation."""
        from argus.decision.calibration import compute_ece
        
        confidences = [0.9, 0.8, 0.7, 0.6]
        outcomes = [1, 1, 1, 0]
        
        ece, data = compute_ece(confidences, outcomes, num_bins=5)
        
        assert 0 <= ece <= 1
        assert "bins" in data
    
    def test_reliability_diagram(self):
        """Test reliability diagram."""
        from argus.decision.calibration import ReliabilityDiagram
        
        diagram = ReliabilityDiagram(num_bins=5)
        
        diagram.add_batch([0.9, 0.8, 0.7], [1, 1, 0])
        
        metrics = diagram.compute()
        
        assert metrics.brier_score >= 0
        assert metrics.num_samples == 3


class TestVoIPlanner:
    """Tests for VoI planner."""
    
    def test_experiment_queue(self):
        """Test experiment queue."""
        from argus.decision.planner import ExperimentQueue
        from argus.decision.eig import ActionCandidate
        
        queue = ExperimentQueue(max_size=3)
        
        a1 = ActionCandidate(id="a1", name="A1", cost=1.0)
        a1.utility = 0.8
        
        a2 = ActionCandidate(id="a2", name="A2", cost=1.0)
        a2.utility = 0.5
        
        queue.push(a1)
        queue.push(a2)
        
        assert len(queue) == 2
        assert queue.peek().id == "a1"  # Higher utility first
