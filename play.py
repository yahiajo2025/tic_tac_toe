"""
Terminal Play Script.
======================
Play against the trained agent in your terminal.

Usage:
    python play.py --model models/q_table.pkl --you x
    python play.py --model models/q_table.pkl --you o
"""

from __future__ import annotations

import argparse

from agents.minimax_agent import MinimaxAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from environment.tictactoe_env import TicTacToeEnv


def print_board_indexed(board):
    """Print board with cell indices (for human reference)."""
    idx = {i: str(i) for i in range(9)}
    symbols = {0: ".", 1: "X", -1: "O"}
    print("\n Board positions:")
    for r in range(3):
        row = " | ".join(f"{idx[3*r+c]}" for c in range(3))
        print(f"  {row}")

    print("\n Current board:")
    for r in range(3):
        row = " | ".join(symbols[board[3 * r + c]] for c in range(3))
        print(f"  {row}")
    print()


def human_action(available: list) -> int:
    """Prompt human for a valid move."""
    while True:
        try:
            choice = int(input(f"  Your move {available}: "))
            if choice in available:
                return choice
            print("  Invalid choice. Try again.")
        except ValueError:
            print("  Please enter a number.")


def play_game(
    agent,
    human_player: int = 1,
    show_q: bool = True,
) -> None:
    """Run one interactive game."""
    env = TicTacToeEnv()
    agent_player = -human_player
    symbol = {1: "X", -1: "O"}

    print(f"\n  You are: {symbol[human_player]}")
    print(f"  Agent is: {symbol[agent_player]}")

    state, _ = env.reset(first_player=1)
    done = False

    while not done:
        print_board_indexed(env.board)
        player = env.current_player
        available = env.available_actions()

        if player == human_player:
            action = human_action(available)
        else:
            action = agent.select_action(state, available, player=agent_player, training=False)
            if show_q and hasattr(agent, "get_q_values"):
                q_vals = agent.get_q_values(state, available, player=agent_player)
                print(f"  AI chose {action} (Q-values: { {k: f'{v:.3f}' for k,v in q_vals.items()} })")
            else:
                print(f"  AI chose {action}")

        state, reward, done, _, info = env.step(action)

    print_board_indexed(env.board)
    winner = info.get("winner")
    if winner == human_player:
        print("  🎉 You win!")
    elif winner == 0:
        print("  🤝 It's a draw!")
    else:
        print("  🤖 AI wins!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe vs RL agent")
    parser.add_argument("--model", type=str, default="models/q_table.pkl")
    parser.add_argument("--you", type=str, default="x", choices=["x", "o"],
                        help="'x' to go first, 'o' to go second")
    parser.add_argument("--opponent", type=str, default="rl",
                        choices=["rl", "minimax", "random"])
    args = parser.parse_args()

    human_player = 1 if args.you == "x" else -1

    if args.opponent == "rl":
        agent = QLearningAgent.load(args.model)
        agent.epsilon = 0.0
    elif args.opponent == "minimax":
        agent = MinimaxAgent()
    else:
        agent = RandomAgent()

    print(f"\n  Opponent: {args.opponent.upper()}")

    while True:
        play_game(agent, human_player=human_player)
        again = input("\n  Play again? (y/n): ").strip().lower()
        if again != "y":
            break
