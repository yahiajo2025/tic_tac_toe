"""
Self-Play Training Script.
===========================
Trains a QLearningAgent via self-play (agent plays both X and O).

Key design:
- Alternates which player goes first every episode.
- Both "sides" share the same Q-table (via state inversion).
- Illegal moves are heavily penalised and the episode continues.
- Saves checkpoints every N episodes.
- Generates training analytics.

Usage:
    python train.py --episodes 200000 --double-q --save models/q_table.pkl
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from agents.q_learning_agent import QLearningAgent
from analytics.plots import TrainingPlotter
from environment.tictactoe_env import TicTacToeEnv


# ---------------------------------------------------------------------------
# Self-play episode
# ---------------------------------------------------------------------------

def run_episode(
    env: TicTacToeEnv,
    agent: QLearningAgent,
    first_player: int = 1,
) -> Dict[str, float]:
    """
    Run a single self-play episode.

    Both X and O are controlled by the same agent. The agent always
    views the board from its own perspective (state inversion for O).

    Returns
    -------
    dict with keys: reward_x, reward_o, winner, illegal_moves
    """
    state, _ = env.reset(first_player=first_player)
    total_reward_x = 0.0
    total_reward_o = 0.0
    illegal_count = 0

    # Buffers for deferred O-update (we need the outcome to assign reward)
    last_o_transition: Tuple | None = None

    done = False
    while not done:
        player = env.current_player
        available = env.available_actions()

        # Illegal move guard: sample until legal (max 9 attempts)
        for attempt in range(9):
            action = agent.select_action(state, available, player=player, training=True)
            next_state, reward, terminated, _, info = env.step(action)

            if info.get("illegal"):
                # Penalise and retry
                agent.update(
                    state, action, TicTacToeEnv.REWARD_ILLEGAL,
                    state, available, False, player=player,
                )
                illegal_count += 1
                # Don't advance state
                continue
            break

        # Accumulate rewards
        if player == 1:
            total_reward_x += reward
        else:
            total_reward_o += reward

        # --- Deferred update for the previous player ---
        # When a move terminates the game, the previous player also needs
        # an update (they just lost / drew).
        if terminated and last_o_transition is not None:
            prev_state, prev_action, prev_player = last_o_transition
            prev_reward = (
                TicTacToeEnv.REWARD_LOSS if info["winner"] == -prev_player
                else TicTacToeEnv.REWARD_DRAW if info["winner"] == 0
                else TicTacToeEnv.REWARD_WIN  # shouldn't happen for the previous player
            )
            agent.update(
                prev_state, prev_action, prev_reward,
                next_state, [], True, player=prev_player,
            )

        # --- Current player update ---
        if not terminated:
            agent.update(
                state, action, reward,
                next_state, env.available_actions(), False, player=player,
            )
            last_o_transition = (state, action, player)
        else:
            agent.update(
                state, action, reward,
                next_state, [], True, player=player,
            )
            last_o_transition = None

        state = next_state
        done = terminated

    return {
        "reward_x": total_reward_x,
        "reward_o": total_reward_o,
        "winner": info.get("winner"),
        "illegal_moves": illegal_count,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    episodes: int = 200_000,
    double_q: bool = True,
    save_path: str = "models/q_table.pkl",
    checkpoint_every: int = 10_000,
    eval_every: int = 5_000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    use_symmetry: bool = True,
    early_stop_win_rate: float = 0.95,
    verbose: bool = True,
) -> QLearningAgent:
    """
    Main training loop.

    Parameters
    ----------
    episodes : int
        Total self-play episodes.
    double_q : bool
        Whether to use Double Q-Learning.
    save_path : str
        Path to save the final model.
    checkpoint_every : int
        Save a checkpoint every N episodes.
    eval_every : int
        Evaluate win-rate every N episodes (vs random).
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon_start / epsilon_end / epsilon_decay : float
        Exploration schedule.
    use_symmetry : bool
        Use canonical state (symmetry reduction).
    early_stop_win_rate : float
        Stop early if win rate vs random exceeds this threshold.
    verbose : bool
        Print progress.

    Returns
    -------
    QLearningAgent
        Trained agent.
    """
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        double_q=double_q,
        use_symmetry=use_symmetry,
    )
    env = TicTacToeEnv()

    # --- History buffers ---
    history: Dict[str, List] = {
        "episode": [],
        "reward_x": [],
        "reward_o": [],
        "epsilon": [],
        "win_rate_vs_random": [],
        "draw_rate_vs_random": [],
        "loss_rate_vs_random": [],
        "q_table_size": [],
        "illegal_moves": [],
    }

    start_time = time.time()

    for ep in tqdm(range(1, episodes + 1), desc="Training", disable=not verbose):
        first_player = 1 if ep % 2 == 0 else -1  # Alternate who goes first
        result = run_episode(env, agent, first_player=first_player)

        agent.decay_epsilon()
        agent.episode_count += 1

        # --- Record basic stats ---
        history["episode"].append(ep)
        history["reward_x"].append(result["reward_x"])
        history["reward_o"].append(result["reward_o"])
        history["epsilon"].append(agent.epsilon)
        history["q_table_size"].append(len(agent.q_a))
        history["illegal_moves"].append(result["illegal_moves"])

        # --- Periodic evaluation vs random ---
        if ep % eval_every == 0:
            from evaluate import evaluate_vs_random
            wr, dr, lr = evaluate_vs_random(agent, n_games=200, verbose=False)
            history["win_rate_vs_random"].append(wr)
            history["draw_rate_vs_random"].append(dr)
            history["loss_rate_vs_random"].append(lr)

            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"\n[Ep {ep:>7}] ε={agent.epsilon:.4f} | "
                    f"WR={wr:.2%} DR={dr:.2%} LR={lr:.2%} | "
                    f"States={len(agent.q_a)} | "
                    f"Time={elapsed:.0f}s"
                )

            # Early stopping
            if wr >= early_stop_win_rate:
                print(f"\n[Early Stop] Win rate {wr:.2%} >= {early_stop_win_rate:.2%}")
                break

        # --- Checkpoint ---
        if ep % checkpoint_every == 0:
            ckpt_path = f"{save_path}.ckpt_{ep}"
            agent.save(ckpt_path)

    # --- Final save ---
    agent.save(save_path)

    # --- Save training history ---
    import pandas as pd
    df = pd.DataFrame({k: v for k, v in history.items() if v})
    df.to_csv("data/training_history.csv", index=False)

    # --- Generate plots ---
    plotter = TrainingPlotter(history)
    plotter.plot_all(save_dir="data/")

    elapsed = time.time() - start_time
    print(f"\n[Done] Training complete in {elapsed:.1f}s | States learned: {len(agent.q_a)}")
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tic-Tac-Toe RL Agent")
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--no-double-q", action="store_true")
    parser.add_argument("--save", type=str, default="models/q_table.pkl")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--early-stop", type=float, default=0.95)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        double_q=not args.no_double_q,
        save_path=args.save,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        use_symmetry=not args.no_symmetry,
        early_stop_win_rate=args.early_stop,
    )
