"""
Tic-Tac-Toe Custom Environment
================================
Gymnasium-like design. Board state is a tuple of 9 integers:
  0 = empty, 1 = Player X, -1 = Player O

State is represented as a canonical tuple for O(1) hashing.
Symmetry reduction is applied via canonical form computation.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple, Dict, Any, List


class TicTacToeEnv:
    """
    Custom Tic-Tac-Toe environment following Gymnasium-like API.

    Attributes
    ----------
    board : list[int]
        9-element flat list. 0=empty, 1=X, -1=O.
    current_player : int
        1 (X) or -1 (O). X always goes first.
    done : bool
        Whether the episode has ended.
    """

    # Reward constants
    REWARD_WIN: float = 1.0
    REWARD_DRAW: float = 0.5
    REWARD_LOSS: float = -1.0
    REWARD_ILLEGAL: float = -10.0
    REWARD_STEP: float = 0.0

    # All winning triples (indices into the 9-cell board)
    WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6),              # diagonals
    )

    # Symmetry group of the board (rotations + reflections)
    # Each transform maps position i → transform[i]
    _SYMMETRIES: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8),  # identity
        (2, 5, 8, 1, 4, 7, 0, 3, 6),  # rotate 90°
        (8, 7, 6, 5, 4, 3, 2, 1, 0),  # rotate 180°
        (6, 3, 0, 7, 4, 1, 8, 5, 2),  # rotate 270°
        (2, 1, 0, 5, 4, 3, 8, 7, 6),  # reflect horizontal
        (6, 7, 8, 3, 4, 5, 0, 1, 2),  # reflect vertical
        (0, 3, 6, 1, 4, 7, 2, 5, 8),  # reflect main diagonal
        (8, 5, 2, 7, 4, 1, 6, 3, 0),  # reflect anti-diagonal
    )

    def __init__(self) -> None:
        self.board: List[int] = [0] * 9
        self.current_player: int = 1
        self.done: bool = False
        self._winner: Optional[int] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, first_player: int = 1) -> Tuple[tuple, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Parameters
        ----------
        first_player : int
            1 for X, -1 for O.

        Returns
        -------
        state : tuple
            The canonical board state (9-element tuple).
        info : dict
            Auxiliary info (empty at reset).
        """
        self.board = [0] * 9
        self.current_player = first_player
        self.done = False
        self._winner = None
        return self._get_state(), {}

    def step(
        self, action: int
    ) -> Tuple[tuple, float, bool, bool, Dict[str, Any]]:
        """
        Apply action and advance the environment.

        Parameters
        ----------
        action : int
            Cell index 0–8 to place the current player's mark.

        Returns
        -------
        next_state : tuple
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        # --- Illegal move ---
        if self.board[action] != 0:
            # Episode continues; agent must retry (handled by caller)
            reward = self.REWARD_ILLEGAL
            return self._get_state(), reward, False, False, {"illegal": True}

        # --- Apply move ---
        self.board[action] = self.current_player

        # --- Check terminal conditions ---
        winner = self.check_winner()
        terminated = False
        reward = self.REWARD_STEP

        if winner == self.current_player:
            reward = self.REWARD_WIN
            terminated = True
            self._winner = winner
        elif winner == -self.current_player:
            reward = self.REWARD_LOSS
            terminated = True
            self._winner = winner
        elif not self.available_actions():
            # Draw
            reward = self.REWARD_DRAW
            terminated = True
            self._winner = 0

        self.done = terminated

        # Switch player
        if not terminated:
            self.current_player = -self.current_player

        info = {
            "winner": self._winner,
            "board": copy.copy(self.board),
            "illegal": False,
        }
        return self._get_state(), reward, terminated, False, info

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the board to stdout or return as string.

        Parameters
        ----------
        mode : str
            'human' prints to stdout; 'ansi' returns a string.
        """
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(3):
            row_str = " | ".join(symbols[self.board[3 * r + c]] for c in range(3))
            rows.append(f" {row_str} ")
        separator = "\n-----------\n"
        board_str = separator.join(rows)

        if mode == "human":
            print(board_str)
            return None
        return board_str

    def available_actions(self) -> List[int]:
        """Return list of empty cell indices."""
        return [i for i, v in enumerate(self.board) if v == 0]

    def check_winner(self) -> Optional[int]:
        """
        Inspect the board for a winner.

        Returns
        -------
        int or None
            1 if X wins, -1 if O wins, 0 if draw (board full, no winner),
            None if game is still ongoing.
        """
        for a, b, c in self.WIN_LINES:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1

        if not self.available_actions():
            return 0  # Draw

        return None

    # ------------------------------------------------------------------
    # State & symmetry helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> tuple:
        """Return current board as an immutable tuple."""
        return tuple(self.board)

    @staticmethod
    def canonical_state(state: tuple) -> tuple:
        """
        Return the canonical (lexicographically smallest) form of a state
        under the 8-element symmetry group (rotations + reflections).

        Used for Q-table size reduction (~8× fewer states on average).
        """
        best = state
        for sym in TicTacToeEnv._SYMMETRIES:
            transformed = tuple(state[sym[i]] for i in range(9))
            if transformed < best:
                best = transformed
        return best

    @staticmethod
    def invert_state(state: tuple) -> tuple:
        """
        Invert the state from the perspective of player O.
        Swaps 1↔-1 so both players share the same Q-table.
        """
        return tuple(-v for v in state)

    def clone(self) -> "TicTacToeEnv":
        """Return a deep copy of the current environment."""
        env = TicTacToeEnv()
        env.board = copy.copy(self.board)
        env.current_player = self.current_player
        env.done = self.done
        env._winner = self._winner
        return env

    def set_board(self, board: List[int], current_player: int) -> None:
        """Manually set board state (useful for minimax and testing)."""
        self.board = list(board)
        self.current_player = current_player
        self.done = self.check_winner() is not None
