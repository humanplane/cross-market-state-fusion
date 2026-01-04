"""
Trading strategies for Polymarket.

Usage:
    from strategies import create_strategy, AVAILABLE_STRATEGIES

    strategy = create_strategy("mean_revert")
    action = strategy.act(state)
"""
from .base import Strategy, MarketState, Action
from .random_strat import RandomStrategy
from .mean_revert import MeanRevertStrategy
from .momentum import MomentumStrategy
from .fade_spike import FadeSpikeStrategy
from .rl_pytorch import RLStrategy  # PyTorch-based PPO (works on Windows/Linux/Mac)
from .gating import GatingStrategy


AVAILABLE_STRATEGIES = [
    "random",
    "mean_revert",
    "momentum",
    "fade_spike",
    "rl",
    "gating",
]


def create_strategy(name: str, **kwargs) -> Strategy:
    """Factory function to create strategies."""
    strategies = {
        "random": RandomStrategy,
        "mean_revert": MeanRevertStrategy,
        "momentum": MomentumStrategy,
        "fade_spike": FadeSpikeStrategy,
        "rl": RLStrategy,
    }

    if name == "gating":
        # Create gating with default experts
        experts = [
            MeanRevertStrategy(),
            MomentumStrategy(),
            FadeSpikeStrategy(),
        ]
        return GatingStrategy(experts, **kwargs)

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")

    return strategies[name](**kwargs)


__all__ = [
    # Base
    "Strategy",
    "MarketState",
    "Action",
    # Strategies
    "RandomStrategy",
    "MeanRevertStrategy",
    "MomentumStrategy",
    "FadeSpikeStrategy",
    "RLStrategy",
    "GatingStrategy",
    # Factory
    "create_strategy",
    "AVAILABLE_STRATEGIES",
]
