"""
Random Agent — Uniform random action selection.
================================================
Baseline opponent for evaluating the RL agent's improvement.
"""

from __future__ import annotations

import random
from typing import List, Optional


class RandomAgent:
    """
    Selects uniformly at random from available legal moves.

    Parameters
    ----------
    player : int
        The token this agent controls (+1 or -1).
    """

    def __init__(self, player: int = -1) -> None:
        self.player = player

    def select_action(
        self,
        state: tuple,
        available_actions: List[int],
        player: Optional[int] = None,
        **kwargs,
    ) -> int:
        """
        Choose a uniformly random legal action.

        Parameters
        ----------
        state : tuple
            Ignored (random policy).
        available_actions : list[int]
            Legal moves.

        Returns
        -------
        int
            Random action.
        """
        if not available_actions:
            raise ValueError("No available actions.")
        return random.choice(available_actions)
