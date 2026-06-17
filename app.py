"""
Streamlit Web Application — Tic-Tac-Toe vs RL Agent.
=====================================================
Modes:
  • Human vs AI
  • AI vs Random  (watch mode)
  • AI vs Minimax (watch mode)

Features:
  • Clean game board with visual feedback.
  • AI confidence display.
  • Q-value heatmap overlay.
  • Win / draw / loss statistics.
  • Restart at any time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

from agents.minimax_agent import MinimaxAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from environment.tictactoe_env import TicTacToeEnv

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tic-Tac-Toe RL",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { max-width: 700px; margin: 0 auto; }
    div[data-testid="stButton"] button {
        width: 100%;
        height: 90px;
        font-size: 2.2rem;
        font-weight: 600;
        border-radius: 12px;
        transition: background 0.15s;
    }
    .stat-card {
        background: #F7F5F0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #D3D1C7;
    }
    .stat-number { font-size: 1.8rem; font-weight: 600; }
    .stat-label  { font-size: 0.8rem; color: #5F5E5A; margin-top: 2px; }
    .winner-banner {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        padding: 14px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(to right, #1D9E75, #EF9F27, #E24B4A);
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/q_table.pkl"

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "board": [0] * 9,
        "current_player": 1,
        "done": False,
        "winner": None,
        "mode": "Human vs AI",
        "human_player": 1,
        "stats": {"wins": 0, "draws": 0, "losses": 0},
        "move_history": [],
        "show_qvalues": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_game() -> None:
    st.session_state.board = [0] * 9
    st.session_state.current_player = 1
    st.session_state.done = False
    st.session_state.winner = None
    st.session_state.move_history = []


# ---------------------------------------------------------------------------
# Agent cache (only load once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_rl_agent() -> Optional[QLearningAgent]:
    if not Path(MODEL_PATH).exists():
        return None
    agent = QLearningAgent.load(MODEL_PATH)
    agent.epsilon = 0.0
    return agent


@st.cache_resource
def load_minimax() -> MinimaxAgent:
    return MinimaxAgent()


# ---------------------------------------------------------------------------
# Game logic helpers
# ---------------------------------------------------------------------------

def get_env_from_state() -> TicTacToeEnv:
    env = TicTacToeEnv()
    env.board = list(st.session_state.board)
    env.current_player = st.session_state.current_player
    env.done = st.session_state.done
    return env


def apply_move(action: int) -> None:
    env = get_env_from_state()
    if env.done or st.session_state.board[action] != 0:
        return

    _, reward, terminated, _, info = env.step(action)
    st.session_state.board = list(env.board)
    st.session_state.current_player = env.current_player
    st.session_state.done = terminated
    st.session_state.winner = info.get("winner")
    st.session_state.move_history.append(action)

    if terminated:
        w = st.session_state.winner
        hp = st.session_state.human_player
        if w == hp:
            st.session_state.stats["wins"] += 1
        elif w == 0:
            st.session_state.stats["draws"] += 1
        else:
            st.session_state.stats["losses"] += 1


def ai_move(agent) -> None:
    env = get_env_from_state()
    if env.done:
        return
    available = env.available_actions()
    if not available:
        return
    player = env.current_player
    action = agent.select_action(
        tuple(env.board), available, player=player, training=False
    )
    apply_move(action)


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

SYMBOLS = {0: "", 1: "✕", -1: "○"}
COLORS = {0: "#FFFFFF", 1: "#E8593C", -1: "#3B8BD4"}


def render_board(agent=None) -> None:
    """Draw the 3×3 board as Streamlit buttons."""
    board = st.session_state.board
    mode = st.session_state.mode
    human_player = st.session_state.human_player

    # Compute Q-values for heatmap
    q_vals = {}
    if agent and st.session_state.show_qvalues and not st.session_state.done:
        env = get_env_from_state()
        available = env.available_actions()
        if available:
            ai_p = -human_player if mode == "Human vs AI" else env.current_player
            q_vals = agent.get_q_values(tuple(board), available, player=ai_p)

    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            idx = row * 3 + col
            cell_val = board[idx]
            symbol = SYMBOLS[cell_val]
            disabled = (
                st.session_state.done
                or cell_val != 0
                or mode != "Human vs AI"
                or st.session_state.current_player != human_player
            )

            # Build label with optional Q-value
            if idx in q_vals:
                label = f"{symbol or '·'}\n{q_vals[idx]:+.2f}"
            else:
                label = symbol or " "

            with cols[col]:
                if st.button(
                    label,
                    key=f"cell_{idx}",
                    disabled=disabled,
                ):
                    apply_move(idx)
                    st.rerun()


def render_confidence(agent, board, available, player) -> None:
    """Show AI confidence bar."""
    if not agent or not available:
        return
    conf = agent.get_confidence(tuple(board), available, player=player)
    pct = int(conf * 100)
    st.markdown(f"**AI confidence:** {pct}%")
    st.progress(conf)


# ---------------------------------------------------------------------------
# Watch mode (AI vs AI)
# ---------------------------------------------------------------------------

def watch_game(agent_x, agent_o, delay: float = 0.5) -> None:
    """Auto-play a game between two agents, rendering each step."""
    env = TicTacToeEnv()
    state, _ = env.reset()

    placeholder = st.empty()

    done = False
    while not done:
        player = env.current_player
        available = env.available_actions()
        agent = agent_x if player == 1 else agent_o
        action = agent.select_action(state, available, player=player, training=False)
        state, _, done, _, info = env.step(action)

        # Display board
        with placeholder.container():
            st.markdown(f"**Turn:** Player {'X' if player == 1 else 'O'} → cell {action}")
            _render_static_board(env.board)

        time.sleep(delay)

    winner = info.get("winner")
    msg = {1: "🎮 Agent X wins!", -1: "🎮 Agent O wins!", 0: "🤝 Draw!"}
    st.success(msg.get(winner, "Game over"))


def _render_static_board(board: list) -> None:
    """Render a read-only board view (for watch mode)."""
    sym = {0: "·", 1: "✕", -1: "○"}
    rows = []
    for r in range(3):
        row = "  |  ".join(sym[board[3 * r + c]] for c in range(3))
        rows.append(f"  {row}")
    st.code("\n".join(rows), language=None)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()
    rl_agent = load_rl_agent()
    minimax_agent = load_minimax()

    # ----------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------
    with st.sidebar:
        st.title("🎮 Tic-Tac-Toe RL")
        st.markdown("*Powered by Double Q-Learning*")
        st.divider()

        mode = st.selectbox(
            "Game mode",
            ["Human vs AI", "AI vs Random", "AI vs Minimax"],
            key="mode_select",
        )
        st.session_state.mode = mode

        if mode == "Human vs AI":
            choice = st.radio("You play as", ["X (first)", "O (second)"])
            st.session_state.human_player = 1 if "X" in choice else -1

        st.session_state.show_qvalues = st.toggle("Show Q-values on board", value=False)

        st.divider()
        if st.button("🔄 Restart Game", use_container_width=True):
            reset_game()
            st.rerun()

        if st.button("🗑️ Reset Stats", use_container_width=True):
            st.session_state.stats = {"wins": 0, "draws": 0, "losses": 0}
            st.rerun()

        st.divider()
        st.subheader("Statistics")
        stats = st.session_state.stats
        total = max(1, sum(stats.values()))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#1D9E75'>{stats['wins']}</div><div class='stat-label'>Wins</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#EF9F27'>{stats['draws']}</div><div class='stat-label'>Draws</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#E24B4A'>{stats['losses']}</div><div class='stat-label'>Losses</div></div>", unsafe_allow_html=True)

        if rl_agent:
            st.divider()
            s = rl_agent.stats()
            st.caption(f"Q-table states: **{s['q_table_size']:,}**")
            st.caption(f"Training episodes: **{s['episode_count']:,}**")
            st.caption(f"Double Q-Learning: **{s['double_q']}**")
        else:
            st.warning("No trained model found. Run `train.py` first.")

    # ----------------------------------------------------------------
    # Main panel
    # ----------------------------------------------------------------
    st.title("Tic-Tac-Toe")

    if mode == "Human vs AI":
        if not rl_agent:
            st.error("No trained model at `models/q_table.pkl`. Run `python train.py` first.")
            return

        # Status
        if st.session_state.done:
            w = st.session_state.winner
            hp = st.session_state.human_player
            if w == hp:
                st.markdown("<div class='winner-banner' style='background:#EAF3DE;color:#3B6D11'>🎉 You win!</div>", unsafe_allow_html=True)
            elif w == 0:
                st.markdown("<div class='winner-banner' style='background:#FAEEDA;color:#633806'>🤝 It's a draw!</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='winner-banner' style='background:#FCEBEB;color:#791F1F'>🤖 AI wins!</div>", unsafe_allow_html=True)
        else:
            cp = st.session_state.current_player
            hp = st.session_state.human_player
            turn_str = "Your turn ✏️" if cp == hp else "AI is thinking... 🤖"
            st.info(turn_str)

        render_board(rl_agent)

        # AI confidence
        if not st.session_state.done:
            env = get_env_from_state()
            available = env.available_actions()
            ai_p = -st.session_state.human_player
            render_confidence(rl_agent, env.board, available, ai_p)

        # Trigger AI move
        if (
            not st.session_state.done
            and st.session_state.current_player != st.session_state.human_player
        ):
            time.sleep(0.3)
            ai_move(rl_agent)
            st.rerun()

    elif mode == "AI vs Random":
        st.info("Watch: RL Agent (X) vs Random Agent (O)")
        if st.button("▶ Run a game"):
            watch_game(rl_agent or minimax_agent, RandomAgent(), delay=0.6)

    elif mode == "AI vs Minimax":
        st.info("Watch: RL Agent (X) vs Minimax Agent (O). A draw = near-perfect play.")
        if st.button("▶ Run a game"):
            watch_game(rl_agent or minimax_agent, minimax_agent, delay=0.8)


if __name__ == "__main__":
    main()
