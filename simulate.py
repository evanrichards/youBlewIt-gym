"""Simulate games between hardcoded strategies to compare performance."""

import random
from collections import defaultdict
from dataclasses import dataclass

from scorer import Scorer
from strategies import BasicStrategy, EvansStrategy, GameAwareStrategy, MomsStrategy, PPOStrategy, RandomStrategy

WINNING_SCORE = 10000
MIN_FIRST_BANK = 1000


@dataclass
class GameResult:
    winner: int  # 0 or 1
    scores: tuple[int, int]
    turns: int


def roll_dice(num_dice: int) -> list[int]:
    return [random.randint(1, 6) for _ in range(num_dice)]


def play_turn(
    strategy: BasicStrategy | MomsStrategy | EvansStrategy | GameAwareStrategy | PPOStrategy | RandomStrategy, total_score: int
) -> int:
    """Play a single turn, return points scored (0 if blown)."""
    current_score = 0
    num_dice = 6

    while True:
        should_roll = strategy.should_roll(0, total_score, num_dice, current_score)

        # Must reach MIN_FIRST_BANK to get on the board
        if total_score == 0 and current_score < MIN_FIRST_BANK:
            should_roll = True

        # Always bank if we can win
        if total_score + current_score >= WINNING_SCORE:
            should_roll = False

        if not should_roll and current_score > 0:
            can_bank = total_score > 0 or current_score >= MIN_FIRST_BANK
            if can_bank:
                return current_score

        dice = roll_dice(num_dice)
        scorer = Scorer(dice)

        if scorer.is_blown():
            return 0

        actions = strategy.actions(dice)
        score = scorer.apply_actions(actions)
        current_score += score

        num_dice = scorer.num_remaining_dice()
        if num_dice == 0:
            num_dice = 6  # Hot dice


def play_game(
    strat1: BasicStrategy | MomsStrategy | EvansStrategy | GameAwareStrategy | PPOStrategy | RandomStrategy,
    strat2: BasicStrategy | MomsStrategy | EvansStrategy | GameAwareStrategy | PPOStrategy | RandomStrategy,
) -> GameResult:
    """Play a full game between two strategies. Returns GameResult."""
    scores = [0, 0]
    strategies = [strat1, strat2]
    turn = 0
    total_turns = 0

    while True:
        points = play_turn(strategies[turn], scores[turn])
        scores[turn] += points
        total_turns += 1

        if scores[turn] >= WINNING_SCORE:
            return GameResult(winner=turn, scores=(scores[0], scores[1]), turns=total_turns)

        turn = 1 - turn


def run_simulation(num_games: int = 10000) -> None:
    """Run head-to-head simulations between all strategy pairs."""
    strategies = {
        "Random": lambda: RandomStrategy(),
        "Basic": lambda: BasicStrategy(),
        "Mom's": lambda: MomsStrategy(),
        "Evan's": lambda: EvansStrategy({1: 300, 2: 300, 3: 350, 4: 400, 5: 500, 6: 600}),
        "Game Aware": lambda: GameAwareStrategy(),
        "PPO": lambda: PPOStrategy(),
    }

    strategy_names = list(strategies.keys())
    results: dict[tuple[str, str], list[GameResult]] = defaultdict(list)

    print(f"Running {num_games} games per matchup...\n")

    # Run all matchups (including self-play)
    for i, name1 in enumerate(strategy_names):
        for name2 in strategy_names[i:]:
            for _ in range(num_games):
                strat1 = strategies[name1]()
                strat2 = strategies[name2]()
                result = play_game(strat1, strat2)
                results[(name1, name2)].append(result)

    # Print results
    print("=" * 60)
    print("HEAD-TO-HEAD RESULTS")
    print("=" * 60)

    for (name1, name2), game_results in results.items():
        p1_wins = sum(1 for r in game_results if r.winner == 0)
        p2_wins = sum(1 for r in game_results if r.winner == 1)
        avg_turns = sum(r.turns for r in game_results) / len(game_results)

        if name1 == name2:
            print(f"\n{name1} vs {name1} (self-play):")
            print(f"  Avg turns per game: {avg_turns:.1f}")
        else:
            p1_pct = 100 * p1_wins / num_games
            p2_pct = 100 * p2_wins / num_games
            print(f"\n{name1} vs {name2}:")
            print(f"  {name1}: {p1_wins:,} wins ({p1_pct:.1f}%)")
            print(f"  {name2}: {p2_wins:,} wins ({p2_pct:.1f}%)")
            print(f"  Avg turns: {avg_turns:.1f}")

    # Overall ranking by win rate
    print("\n" + "=" * 60)
    print("OVERALL RANKING (by avg win rate across matchups)")
    print("=" * 60)

    win_rates: dict[str, list[float]] = defaultdict(list)
    for (name1, name2), game_results in results.items():
        if name1 == name2:
            continue
        p1_wins = sum(1 for r in game_results if r.winner == 0)
        p2_wins = sum(1 for r in game_results if r.winner == 1)
        win_rates[name1].append(p1_wins / num_games)
        win_rates[name2].append(p2_wins / num_games)

    avg_win_rates = {name: sum(rates) / len(rates) for name, rates in win_rates.items()}
    ranked = sorted(avg_win_rates.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, rate) in enumerate(ranked, 1):
        print(f"  {rank}. {name}: {100 * rate:.1f}% avg win rate")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate strategy matchups")
    parser.add_argument("-n", "--num-games", type=int, default=10000, help="Games per matchup")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    run_simulation(args.num_games)
