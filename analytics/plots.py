"""
Training Analytics — Chart generation.
========================================
Generates 4 publication-quality training charts:
  1. Reward curve (rolling average).
  2. Win rate vs Random over training.
  3. Epsilon (exploration) decay curve.
  4. Q-table growth (states discovered).

All saved to PNG and optionally displayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


class TrainingPlotter:
    """
    Wraps training history and renders analytics charts.

    Parameters
    ----------
    history : dict
        Dictionary of lists from the training loop.
    """

    # Consistent colour palette
    PALETTE = {
        "x": "#E8593C",
        "o": "#3B8BD4",
        "win": "#1D9E75",
        "draw": "#EF9F27",
        "loss": "#E24B4A",
        "epsilon": "#7F77DD",
        "states": "#888780",
    }

    def __init__(self, history: Dict[str, List]) -> None:
        self.history = history
        self._setup_style()

    @staticmethod
    def _setup_style() -> None:
        """Apply a clean, minimal plot style."""
        plt.rcParams.update({
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#F7F5F0",
            "axes.grid": True,
            "grid.color": "#DDDBD5",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "medium",
            "legend.frameon": False,
        })

    def _rolling(self, data: List[float], window: int = 500) -> np.ndarray:
        """Compute rolling average."""
        arr = np.array(data, dtype=float)
        return pd.Series(arr).rolling(window, min_periods=1).mean().values

    # ------------------------------------------------------------------
    # Individual charts
    # ------------------------------------------------------------------

    def plot_reward_curve(self, ax: plt.Axes, window: int = 500) -> None:
        """Reward (rolling average) for X and O players over episodes."""
        eps = self.history.get("episode", [])
        rx = self._rolling(self.history.get("reward_x", []), window)
        ro = self._rolling(self.history.get("reward_o", []), window)

        ax.plot(eps, rx, color=self.PALETTE["x"], linewidth=1.2, label="Player X reward")
        ax.plot(eps, ro, color=self.PALETTE["o"], linewidth=1.2, label="Player O reward", alpha=0.8)
        ax.axhline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Reward (rolling)")
        ax.set_title(f"Reward Curve  (window={window})")
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    def plot_win_rate_curve(self, ax: plt.Axes) -> None:
        """Win / draw / loss rate vs random agent over training."""
        wr = self.history.get("win_rate_vs_random", [])
        dr = self.history.get("draw_rate_vs_random", [])
        lr = self.history.get("loss_rate_vs_random", [])
        eval_eps = self.history.get("eval_episodes", [])

        if not wr:
            ax.text(0.5, 0.5, "No evaluation data", transform=ax.transAxes, ha="center")
            return

        # Reconstruct evaluation episode points
        n = len(wr)
        total_eps = max(self.history.get("episode", [1]))
        eval_eps = [int((i + 1) * total_eps / n) for i in range(n)]

        ax.plot(eval_eps, wr, color=self.PALETTE["win"], linewidth=1.5, marker="o",
                markersize=3, label="Win %")
        ax.plot(eval_eps, dr, color=self.PALETTE["draw"], linewidth=1.5, marker="s",
                markersize=3, label="Draw %")
        ax.plot(eval_eps, lr, color=self.PALETTE["loss"], linewidth=1.5, marker="^",
                markersize=3, label="Loss %")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate")
        ax.set_title("Win / Draw / Loss Rate vs Random")
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    def plot_epsilon_curve(self, ax: plt.Axes) -> None:
        """Epsilon (exploration rate) decay over training."""
        eps = self.history.get("episode", [])
        epsilon = self.history.get("epsilon", [])

        ax.plot(eps, epsilon, color=self.PALETTE["epsilon"], linewidth=1.4)
        ax.fill_between(eps, epsilon, alpha=0.15, color=self.PALETTE["epsilon"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon ε")
        ax.set_title("Exploration Decay")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    def plot_q_table_growth(self, ax: plt.Axes) -> None:
        """Number of unique Q-table states discovered."""
        eps = self.history.get("episode", [])
        sizes = self.history.get("q_table_size", [])

        ax.plot(eps, sizes, color=self.PALETTE["states"], linewidth=1.4)
        ax.fill_between(eps, sizes, alpha=0.1, color=self.PALETTE["states"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Unique States")
        ax.set_title("Q-Table Growth")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))

    # ------------------------------------------------------------------
    # Master dashboard
    # ------------------------------------------------------------------

    def plot_all(self, save_dir: str = "data/", show: bool = False) -> None:
        """
        Render all 4 charts in a 2×2 grid and save to PNG.

        Parameters
        ----------
        save_dir : str
            Directory to write PNGs.
        show : bool
            If True, also call plt.show().
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Tic-Tac-Toe RL Training Analytics", fontsize=15, fontweight="medium", y=1.01)

        self.plot_reward_curve(axes[0, 0])
        self.plot_win_rate_curve(axes[0, 1])
        self.plot_epsilon_curve(axes[1, 0])
        self.plot_q_table_growth(axes[1, 1])

        plt.tight_layout()
        out = Path(save_dir) / "training_dashboard.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved dashboard → {out}")

        if show:
            plt.show()
        plt.close(fig)

        # Also save individual charts
        self._save_individual(save_dir)

    def _save_individual(self, save_dir: str) -> None:
        """Save each chart individually."""
        charts = [
            ("reward_curve.png", self.plot_reward_curve),
            ("win_rate_curve.png", self.plot_win_rate_curve),
            ("epsilon_curve.png", self.plot_epsilon_curve),
            ("q_table_growth.png", self.plot_q_table_growth),
        ]
        for fname, plot_fn in charts:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_fn(ax)
            plt.tight_layout()
            fig.savefig(Path(save_dir) / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
