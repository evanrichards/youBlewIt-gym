"""Interactive You Blew It game using Streamlit, powered by the gym environment."""

import random
import time

import gymnasium as gym
import pandas as pd
import streamlit as st

import gym_env  # noqa: F401 - registers environments
from gym_env.you_blew_it import YouBlewItEnv
from scorer import Scorer
from strategies import BasicStrategy, EvansStrategy, MomsStrategy, RandomStrategy

# Strategy options with descriptions
STRATEGIES = {
    "Basic": ("Simple strategy - stops with 2 or fewer dice", lambda: BasicStrategy()),
    "Mom's": ("Conservative - adjusts based on score thresholds", lambda: MomsStrategy()),
    "Evan's": (
        "Tuned thresholds per remaining dice count",
        lambda: EvansStrategy({1: 300, 2: 300, 3: 350, 4: 400, 5: 500, 6: 600}),
    ),
    "Random": ("Random legal moves - chaotic baseline", lambda: RandomStrategy()),
}

# Opponent display names and colors
OPPONENT_NAMES = ["Opponent 1", "Opponent 2", "Opponent 3"]
OPPONENT_ICONS = ["🤖", "👾", "🎰"]

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
        reset_game(["Basic"], spectator_mode=False)


def reset_game(
    strategy_names: list[str],
    spectator_mode: bool = False,
    player_strategy: str | None = None,
) -> None:
    """Reset all game state for a new game with multiple opponents."""
    st.session_state.initialized = True
    st.session_state.env: YouBlewItEnv = gym.make("YouBlewIt-v1").unwrapped  # type: ignore[assignment]
    st.session_state.env.reset()
    st.session_state.player_score = 0

    # Spectator mode settings
    st.session_state.spectator_mode = spectator_mode
    st.session_state.player_strategy_name = player_strategy
    if spectator_mode and player_strategy:
        st.session_state.player_strategy = STRATEGIES[player_strategy][1]()
    else:
        st.session_state.player_strategy = None

    # Multiple opponents
    st.session_state.num_opponents = len(strategy_names)
    st.session_state.strategy_names = strategy_names
    st.session_state.opponent_scores = [0] * len(strategy_names)
    st.session_state.opponent_strategies = [STRATEGIES[name][1]() for name in strategy_names]

    st.session_state.current_turn = "player"
    st.session_state.current_opponent_idx = 0
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = "Your turn! Roll the dice." if not spectator_mode else "Watch the game!"

    st.session_state.blew_it = False
    st.session_state.blew_it_dice = []
    st.session_state.blew_it_lost_score = 0

    st.session_state.opponent_turn_done = False
    st.session_state.opponent_turn_log = []

    # Player turn log for spectator mode
    st.session_state.player_turn_done = False
    st.session_state.player_turn_log = []

    # Score history for chart
    player_label = f"You ({player_strategy})" if spectator_mode and player_strategy else "You"
    st.session_state.score_history = {"turn": [0], player_label: [0]}
    st.session_state.player_chart_label = player_label
    for i in range(len(strategy_names)):
        st.session_state.score_history[f"{OPPONENT_NAMES[i]} ({strategy_names[i]})"] = [0]
    st.session_state.turn_number = 0


def get_env() -> YouBlewItEnv:
    """Get the current environment."""
    return st.session_state.env


def record_scores() -> None:
    """Record current scores to history after a turn."""
    st.session_state.turn_number += 1
    st.session_state.score_history["turn"].append(st.session_state.turn_number)
    player_label = st.session_state.player_chart_label
    st.session_state.score_history[player_label].append(st.session_state.player_score)
    for i, name in enumerate(st.session_state.strategy_names):
        key = f"{OPPONENT_NAMES[i]} ({name})"
        st.session_state.score_history[key].append(st.session_state.opponent_scores[i])


def roll_dice(num_dice: int) -> list[int]:
    """Roll dice for opponent (env handles player rolls)."""
    return [random.randint(1, 6) for _ in range(num_dice)]


def do_action(action: int, pre_roll_score: int = 0) -> None:
    """Execute an action in the environment."""
    env = get_env()
    obs, reward, terminated, truncated, info = env.step(action)

    if "reason" in info:
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
            # Start opponent turns
            st.session_state.current_turn = "opponent"
            st.session_state.current_opponent_idx = 0
    elif action == 9:  # Roll
        if env.blown:
            st.session_state.blew_it = True
            st.session_state.blew_it_dice = list(env.dice)
            st.session_state.blew_it_lost_score = pre_roll_score
        else:
            st.session_state.message = "Take at least one scoring die, then roll again or bank."
    else:  # Took dice
        st.session_state.message = f"Turn score: {env.unbanked_score}"


def play_opponent_turn(opponent_idx: int) -> list[str]:
    """Play a single opponent's turn."""
    strategy = st.session_state.opponent_strategies[opponent_idx]
    opponent_score = st.session_state.opponent_scores[opponent_idx]
    turn_log = []
    current_score = 0
    num_dice = 6

    while True:
        should_roll = strategy.should_roll(0, opponent_score, num_dice, current_score)

        if opponent_score == 0 and current_score < MIN_FIRST_BANK:
            should_roll = True

        if opponent_score + current_score >= WINNING_SCORE:
            should_roll = False

        if not should_roll and current_score > 0:
            can_bank = opponent_score > 0 or current_score >= MIN_FIRST_BANK
            if can_bank:
                st.session_state.opponent_scores[opponent_idx] += current_score
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


def play_player_turn_auto() -> list[str]:
    """Play the player's turn automatically using their strategy (spectator mode)."""
    strategy = st.session_state.player_strategy
    turn_log = []
    current_score = 0
    num_dice = 6

    while True:
        should_roll = strategy.should_roll(
            0, st.session_state.player_score, num_dice, current_score
        )

        if st.session_state.player_score == 0 and current_score < MIN_FIRST_BANK:
            should_roll = True

        if st.session_state.player_score + current_score >= WINNING_SCORE:
            should_roll = False

        if not should_roll and current_score > 0:
            can_bank = st.session_state.player_score > 0 or current_score >= MIN_FIRST_BANK
            if can_bank:
                st.session_state.player_score += current_score
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
    """Render the scoreboard with multiple opponents."""
    env = get_env()
    num_opponents = st.session_state.num_opponents
    spectator = st.session_state.spectator_mode

    # Create columns: player + opponents
    cols = st.columns(1 + num_opponents)

    with cols[0]:
        if spectator:
            player_name = st.session_state.player_strategy_name
            label = f"🎯 {player_name}"
            st.metric(label, f"{st.session_state.player_score:,}")
        else:
            delta = f"+{env.unbanked_score}" if env.unbanked_score > 0 else None
            st.metric("You", f"{st.session_state.player_score:,}", delta=delta)

    for i in range(num_opponents):
        with cols[i + 1]:
            name = st.session_state.strategy_names[i]
            score = st.session_state.opponent_scores[i]
            st.metric(f"{OPPONENT_ICONS[i]} {name}", f"{score:,}")

    # Progress bars
    player_label = st.session_state.player_chart_label
    st.progress(min(st.session_state.player_score / WINNING_SCORE, 1.0), text=player_label)
    for i in range(num_opponents):
        name = st.session_state.strategy_names[i]
        score = st.session_state.opponent_scores[i]
        st.progress(min(score / WINNING_SCORE, 1.0), text=f"{OPPONENT_ICONS[i]} {name}")

    # Score history chart
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
        st.header("Game Mode")

        spectator_mode = st.toggle("Spectator Mode", value=False, key="spectator_toggle")

        if spectator_mode:
            st.caption("Watch AI strategies compete against each other")
            strategy_options = list(STRATEGIES.keys())
            player_strat = st.selectbox(
                "🎯 Player 1 Strategy",
                strategy_options,
                index=0,
                key="player_strategy_selector",
            )
            description, _ = STRATEGIES[player_strat]
            st.caption(description)
        else:
            player_strat = None

        st.divider()
        st.header("Opponents")

        # Number of opponents
        num_opponents = st.selectbox(
            "Number of opponents",
            [1, 2, 3],
            index=0,
            key="num_opponents_selector",
        )

        # Strategy selection for each opponent
        strategy_options = list(STRATEGIES.keys())
        selected_strategies: list[str] = []

        for i in range(num_opponents):
            selected = st.selectbox(
                f"{OPPONENT_ICONS[i]} Opponent {i + 1}",
                strategy_options,
                index=0,
                key=f"strategy_selector_{i}",
            )
            selected_strategies.append(selected)
            # Show description
            description, _ = STRATEGIES[selected]
            st.caption(description)

        if st.button("New Game", type="primary", use_container_width=True):
            reset_game(selected_strategies, spectator_mode, player_strat)
            st.rerun()

    # Game over screen
    if st.session_state.game_over:
        winner = st.session_state.winner
        spectator = st.session_state.spectator_mode
        if winner == "player":
            if spectator:
                player_name = st.session_state.player_strategy_name
                st.success(f"🎯 {player_name} wins with {st.session_state.player_score:,} points!")
            else:
                st.balloons()
                st.success(f"🎉 You win with {st.session_state.player_score:,} points!")
        else:
            # winner is opponent index
            idx = winner
            name = st.session_state.strategy_names[idx]
            score = st.session_state.opponent_scores[idx]
            if spectator:
                st.success(f"{OPPONENT_ICONS[idx]} {name} wins with {score:,} points!")
            else:
                st.error(f"😢 {OPPONENT_ICONS[idx]} {name} wins with {score:,} points!")

        if st.button("Play Again", type="primary"):
            reset_game(
                st.session_state.strategy_names,
                st.session_state.spectator_mode,
                st.session_state.player_strategy_name,
            )
            st.rerun()
        return

    render_scoreboard()
    st.divider()

    # Handle "blew it" pause screen (only in non-spectator mode)
    if st.session_state.blew_it and not st.session_state.spectator_mode:
        st.error("💥 You Blew It!")
        if st.session_state.blew_it_dice:
            render_dice(st.session_state.blew_it_dice, "You rolled")
        if st.session_state.blew_it_lost_score > 0:
            st.warning(f"Lost {st.session_state.blew_it_lost_score} unbanked points!")
        else:
            st.info("No points lost this roll.")

        if st.button("Continue to Opponents' Turn", type="primary", use_container_width=True):
            st.session_state.blew_it = False
            st.session_state.blew_it_dice = []
            st.session_state.blew_it_lost_score = 0
            st.session_state.current_turn = "opponent"
            st.session_state.current_opponent_idx = 0
            st.rerun()
        return

    # Handle opponent turns
    if st.session_state.current_turn == "opponent" and not st.session_state.opponent_turn_done:
        idx = st.session_state.current_opponent_idx
        name = st.session_state.strategy_names[idx]
        with st.spinner(f"{OPPONENT_ICONS[idx]} {name} is playing..."):
            time.sleep(0.5)
            turn_log = play_opponent_turn(idx)
            st.session_state.opponent_turn_log = turn_log
            st.session_state.opponent_turn_done = True
            record_scores()
            st.rerun()

    # Handle opponent turn done - show results
    if st.session_state.opponent_turn_done:
        idx = st.session_state.current_opponent_idx
        name = st.session_state.strategy_names[idx]
        score = st.session_state.opponent_scores[idx]

        st.subheader(f"{OPPONENT_ICONS[idx]} {name}'s Turn")
        for entry in st.session_state.opponent_turn_log:
            st.write(entry)

        # Check for win
        if score >= WINNING_SCORE:
            st.session_state.game_over = True
            st.session_state.winner = idx
            st.rerun()

        st.divider()

        # Determine next action
        next_idx = idx + 1
        spectator = st.session_state.spectator_mode
        if next_idx < st.session_state.num_opponents:
            # More opponents to play
            next_name = st.session_state.strategy_names[next_idx]
            btn_label = f"Continue to {OPPONENT_ICONS[next_idx]} {next_name}'s Turn"
        else:
            # All opponents done, back to player
            if spectator:
                player_name = st.session_state.player_strategy_name
                btn_label = f"Continue to 🎯 {player_name}'s Turn"
            else:
                btn_label = "Continue to Your Turn"

        if st.button(btn_label, type="primary", use_container_width=True):
            st.session_state.opponent_turn_done = False
            st.session_state.opponent_turn_log = []

            if next_idx < st.session_state.num_opponents:
                # Next opponent
                st.session_state.current_opponent_idx = next_idx
            else:
                # Back to player
                st.session_state.current_turn = "player"
                st.session_state.current_opponent_idx = 0
                if not spectator:
                    env = get_env()
                    env.reset()
                    env.score = st.session_state.player_score
                    st.session_state.message = "Your turn! Roll the dice."

            st.rerun()
        return

    # Player's turn
    spectator = st.session_state.spectator_mode

    # Spectator mode: auto-play player turn
    if spectator:
        if (
            st.session_state.current_turn == "player"
            and not st.session_state.player_turn_done
        ):
            player_name = st.session_state.player_strategy_name
            with st.spinner(f"🎯 {player_name} is playing..."):
                time.sleep(0.5)
                turn_log = play_player_turn_auto()
                st.session_state.player_turn_log = turn_log
                st.session_state.player_turn_done = True
                record_scores()
                st.rerun()

        # Show player turn results
        if st.session_state.player_turn_done:
            player_name = st.session_state.player_strategy_name
            st.subheader(f"🎯 {player_name}'s Turn")
            for entry in st.session_state.player_turn_log:
                st.write(entry)

            # Check for win
            if st.session_state.player_score >= WINNING_SCORE:
                st.session_state.game_over = True
                st.session_state.winner = "player"
                st.rerun()

            st.divider()
            next_name = st.session_state.strategy_names[0]
            btn_label = f"Continue to {OPPONENT_ICONS[0]} {next_name}'s Turn"

            if st.button(btn_label, type="primary", use_container_width=True):
                st.session_state.player_turn_done = False
                st.session_state.player_turn_log = []
                st.session_state.current_turn = "opponent"
                st.session_state.current_opponent_idx = 0
                st.rerun()
        return

    # Non-spectator: manual player turn
    env = get_env()
    st.subheader("Your Turn")
    st.info(st.session_state.message)

    if any(d != 0 for d in env.dice):
        render_dice(env.dice, "Current Roll")

    legal = env.legal_actions()
    mask = env.action_masks()

    col1, col2 = st.columns(2)

    with col1:
        if mask[9]:
            if st.button("🎲 Roll Dice", type="primary", use_container_width=True):
                do_action(9, pre_roll_score=env.unbanked_score)
                st.rerun()

    with col2:
        if mask[0]:
            bank_label = f"💰 Bank {env.unbanked_score} points"
            if st.button(bank_label, type="secondary", use_container_width=True):
                do_action(0)
                st.rerun()
        elif env.unbanked_score > 0 and st.session_state.player_score == 0:
            need = MIN_FIRST_BANK - env.unbanked_score
            if need > 0:
                st.warning(f"Need {need} more to bank")

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
