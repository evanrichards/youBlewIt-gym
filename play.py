"""Interactive You Blew It game using Streamlit, powered by the gym environment."""

import time

import gymnasium as gym
import pandas as pd
import streamlit as st

import gym_env  # noqa: F401 - registers environments
from gym_env.you_blew_it import YouBlewItEnv
from scorer import Scorer
from strategies import BasicStrategy, EvansStrategy, MomsStrategy

# Strategy options with descriptions
STRATEGIES = {
    "Basic": ("Simple strategy - stops with 2 or fewer dice", lambda: BasicStrategy()),
    "Mom's": ("Conservative - adjusts based on score thresholds", lambda: MomsStrategy()),
    "Evan's": ("Tuned thresholds per remaining dice count", lambda: EvansStrategy({
        1: 300, 2: 300, 3: 350, 4: 400, 5: 500, 6: 600
    })),
}

# Game constants
WINNING_SCORE = 10000
MIN_FIRST_BANK = 800

# Dice face representations
DICE_FACES = {
    0: " ",
    1: "⚀",
    2: "⚁",
    3: "⚂",
    4: "⚃",
    5: "⚄",
    6: "⚅",
}

# Action names for display
ACTION_NAMES = {
    0: ("Bank", "💰"),
    1: ("Take three 1s (+1000)", "🎯"),
    2: ("Take three 2s (+200)", "🎲"),
    3: ("Take three 3s (+300)", "🎲"),
    4: ("Take three 4s (+400)", "🎲"),
    5: ("Take three 5s (+500)", "🎲"),
    6: ("Take three 6s (+600)", "🎲"),
    7: ("Take a 5 (+50)", "5️⃣"),
    8: ("Take a 1 (+100)", "1️⃣"),
    9: ("Roll", "🎲"),
}


def init_game_state() -> None:
    """Initialize or reset the game state."""
    if "initialized" not in st.session_state:
        reset_game()


def reset_game(strategy_name: str = "Basic") -> None:
    """Reset all game state for a new game."""
    st.session_state.initialized = True
    st.session_state.env: YouBlewItEnv = gym.make("YouBlewIt-v1").unwrapped  # type: ignore[assignment]
    st.session_state.env.reset()
    st.session_state.player_score = 0
    st.session_state.opponent_score = 0
    st.session_state.current_turn = "player"
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = "Your turn! Roll the dice."
    st.session_state.strategy_name = strategy_name
    _, strategy_factory = STRATEGIES[strategy_name]
    st.session_state.opponent_strategy = strategy_factory()
    st.session_state.blew_it = False  # Pause state when player blows it
    st.session_state.blew_it_dice = []  # Store dice that caused the blow
    st.session_state.blew_it_lost_score = 0  # Score that was lost
    st.session_state.opponent_turn_done = False  # Pause after opponent's turn
    st.session_state.opponent_turn_log = []  # Log of opponent's turn actions
    st.session_state.score_history = {"turn": [0], "You": [0], "Opponent": [0]}  # Track scores over time
    st.session_state.turn_number = 0


def get_env() -> YouBlewItEnv:
    """Get the current environment."""
    return st.session_state.env


def record_scores() -> None:
    """Record current scores to history after a turn."""
    st.session_state.turn_number += 1
    st.session_state.score_history["turn"].append(st.session_state.turn_number)
    st.session_state.score_history["You"].append(st.session_state.player_score)
    st.session_state.score_history["Opponent"].append(st.session_state.opponent_score)


def roll_dice(num_dice: int) -> list[int]:
    """Roll dice for opponent (env handles player rolls)."""
    import random
    return [random.randint(1, 6) for _ in range(num_dice)]


def do_action(action: int, pre_roll_dice: list[int] | None = None, pre_roll_score: int = 0) -> None:
    """Execute an action in the environment."""
    env = get_env()
    obs, reward, terminated, truncated, info = env.step(action)

    if "reason" in info:
        # Illegal move - shouldn't happen if we use legal_actions
        st.session_state.message = f"Illegal: {info['reason']}"
        return

    if action == 0:  # Bank
        st.session_state.player_score = env.score
        st.session_state.message = f"Banked! Total score: {env.score}"
        record_scores()
        if env.score >= WINNING_SCORE:
            st.session_state.game_over = True
            st.session_state.winner = "player"
        else:
            st.session_state.current_turn = "opponent"
    elif action == 9:  # Roll
        if env.blown:
            # Store state for the "blew it" pause screen
            st.session_state.blew_it = True
            st.session_state.blew_it_dice = list(env.dice)
            st.session_state.blew_it_lost_score = pre_roll_score
        else:
            st.session_state.message = "Take at least one scoring die, then roll again or bank."
    else:  # Took dice
        st.session_state.message = f"Turn score: {env.unbanked_score}"


def play_opponent_turn() -> list[str]:
    """Play the opponent's turn using BasicStrategy."""
    strategy = st.session_state.opponent_strategy
    turn_log = []
    current_score = 0
    num_dice = 6

    while True:
        should_roll = strategy.should_roll(
            0, st.session_state.opponent_score, num_dice, current_score
        )

        if st.session_state.opponent_score == 0 and current_score < MIN_FIRST_BANK:
            should_roll = True

        if st.session_state.opponent_score + current_score >= WINNING_SCORE:
            should_roll = False

        if not should_roll and current_score > 0:
            can_bank_opp = st.session_state.opponent_score > 0 or current_score >= MIN_FIRST_BANK
            if can_bank_opp:
                st.session_state.opponent_score += current_score
                turn_log.append(f"Banked {current_score} points!")
                break

        dice = roll_dice(num_dice)
        turn_log.append(f"Rolled: {' '.join(DICE_FACES[d] for d in dice)}")

        scorer = Scorer(dice)
        if scorer.is_blown():
            turn_log.append("Blew it!")
            break

        actions = strategy.actions(dice)
        score = scorer.apply_actions(actions)
        current_score += score
        turn_log.append(f"Took +{score} (turn total: {current_score})")

        num_dice = scorer.num_remaining_dice()
        if num_dice == 0:
            num_dice = 6
            turn_log.append("Hot dice! Rolling all 6 again.")

    return turn_log


def render_dice(dice: list[int], label: str = "Dice") -> None:
    """Render dice visually, filtering out zeros."""
    visible_dice = [d for d in dice if d != 0]
    if not visible_dice:
        return
    st.write(f"**{label}:**")
    dice_str = " ".join(DICE_FACES[d] for d in visible_dice)
    st.markdown(
        f"<h1 style='font-size: 4rem; letter-spacing: 0.5rem;'>{dice_str}</h1>",
        unsafe_allow_html=True,
    )


def render_scoreboard() -> None:
    """Render the scoreboard."""
    env = get_env()
    col1, col2 = st.columns(2)
    with col1:
        delta = f"+{env.unbanked_score}" if env.unbanked_score > 0 else None
        st.metric("Your Score", st.session_state.player_score, delta=delta)
    with col2:
        st.metric("Opponent Score", st.session_state.opponent_score)

    st.progress(min(st.session_state.player_score / WINNING_SCORE, 1.0), text="You")
    st.progress(min(st.session_state.opponent_score / WINNING_SCORE, 1.0), text="Opponent")

    # Show score history chart if there's data
    if len(st.session_state.score_history["turn"]) > 1:
        df = pd.DataFrame(st.session_state.score_history)
        df = df.set_index("turn")
        st.line_chart(df, height=200)


def main() -> None:
    st.set_page_config(page_title="You Blew It!", page_icon="🎲", layout="centered")
    st.title("🎲 You Blew It!")
    st.caption(f"First to {WINNING_SCORE:,} wins! Need {MIN_FIRST_BANK}+ to get on the board.")

    init_game_state()

    with st.sidebar:
        st.header("Scoring")
        st.markdown("""
        **Three of a kind:**
        - Three 1s = 1,000
        - Three 2s = 200
        - Three 3s = 300
        - Three 4s = 400
        - Three 5s = 500
        - Three 6s = 600

        **Single dice:**
        - Each 1 = 100
        - Each 5 = 50

        **Rules:**
        - Must keep at least one scoring die per roll
        - "Hot dice": use all 6, roll again!
        - Bank to keep your points
        - Roll with no scoring dice = lose turn points
        """)

        st.divider()
        st.header("Opponent")

        # Get current strategy name, default to Basic
        current_strategy = getattr(st.session_state, "strategy_name", "Basic")

        strategy_options = list(STRATEGIES.keys())
        selected_strategy = st.selectbox(
            "Strategy",
            strategy_options,
            index=strategy_options.index(current_strategy),
            key="strategy_selector",
        )

        # Show strategy description
        description, _ = STRATEGIES[selected_strategy]
        st.caption(description)

        if st.button("New Game", type="primary", use_container_width=True):
            reset_game(selected_strategy)
            st.rerun()

    if st.session_state.game_over:
        if st.session_state.winner == "player":
            st.balloons()
            st.success(f"🎉 You win with {st.session_state.player_score:,} points!")
        else:
            st.error(f"😢 Opponent wins with {st.session_state.opponent_score:,} points!")

        if st.button("Play Again", type="primary"):
            reset_game(st.session_state.strategy_name)
            st.rerun()
        return

    render_scoreboard()
    st.divider()

    # Handle "blew it" pause screen
    if st.session_state.blew_it:
        st.error("💥 You Blew It!")
        if st.session_state.blew_it_dice:
            render_dice(st.session_state.blew_it_dice, "You rolled")
        if st.session_state.blew_it_lost_score > 0:
            st.warning(f"Lost {st.session_state.blew_it_lost_score} unbanked points!")
        else:
            st.info("No points lost this roll.")

        if st.button("Continue to Opponent's Turn", type="primary", use_container_width=True):
            st.session_state.blew_it = False
            st.session_state.blew_it_dice = []
            st.session_state.blew_it_lost_score = 0
            st.session_state.current_turn = "opponent"
            st.rerun()
        return

    # Handle opponent turn - play it
    if st.session_state.current_turn == "opponent" and not st.session_state.opponent_turn_done:
        with st.spinner("🤖 Opponent is playing..."):
            time.sleep(0.5)
            turn_log = play_opponent_turn()
            st.session_state.opponent_turn_log = turn_log
            st.session_state.opponent_turn_done = True
            record_scores()
            st.rerun()

    # Handle opponent turn done - show results and wait for confirmation
    if st.session_state.opponent_turn_done:
        st.subheader("🤖 Opponent's Turn")

        # Show the turn log
        for entry in st.session_state.opponent_turn_log:
            st.write(entry)

        # Check for opponent win
        if st.session_state.opponent_score >= WINNING_SCORE:
            st.session_state.game_over = True
            st.session_state.winner = "opponent"
            st.rerun()

        st.divider()
        if st.button("Continue to Your Turn", type="primary", use_container_width=True):
            # Reset player env for new turn
            env = get_env()
            env.reset()
            env.score = st.session_state.player_score  # Preserve player's banked score

            st.session_state.opponent_turn_done = False
            st.session_state.opponent_turn_log = []
            st.session_state.current_turn = "player"
            st.session_state.message = "Your turn! Roll the dice."
            st.rerun()
        return

    # Player's turn
    env = get_env()
    st.subheader("Your Turn")
    st.info(st.session_state.message)

    # Show dice
    if any(d != 0 for d in env.dice):
        render_dice(env.dice, "Current Roll")

    # Get legal actions from environment
    legal = env.legal_actions()
    mask = env.action_masks()

    # Organize actions
    col1, col2 = st.columns(2)

    with col1:
        # Roll button
        if mask[9]:
            if st.button("🎲 Roll Dice", type="primary", use_container_width=True):
                do_action(9, pre_roll_score=env.unbanked_score)
                st.rerun()

    with col2:
        # Bank button
        if mask[0]:
            bank_label = f"💰 Bank {env.unbanked_score} points"
            if st.button(bank_label, type="secondary", use_container_width=True):
                do_action(0)
                st.rerun()
        elif env.unbanked_score > 0 and st.session_state.player_score == 0:
            need = MIN_FIRST_BANK - env.unbanked_score
            if need > 0:
                st.warning(f"Need {need} more to bank")

    # Scoring actions (1-8)
    scoring_actions = [a for a in legal if 1 <= a <= 8]
    if scoring_actions:
        st.write("**Available moves:**")
        cols = st.columns(min(len(scoring_actions), 3))
        for i, action in enumerate(scoring_actions):
            name, emoji = ACTION_NAMES[action]
            with cols[i % 3]:
                if st.button(f"{emoji} {name}", key=f"action_{action}", use_container_width=True):
                    do_action(action)
                    st.rerun()


if __name__ == "__main__":
    main()
