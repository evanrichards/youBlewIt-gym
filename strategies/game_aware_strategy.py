"""Game-aware strategy based on exact dynamic programming over all roll outcomes.

This strategy implements optimal banking thresholds and endgame racing for Farkle
with singles (1/5) + triples scoring, hot dice, and 10k first-past-the-line rules.
"""

from scorer import Scorer


class GameAwareStrategy:
    """Dynamic strategy that adjusts based on game state and opponent positions."""

    # Base risk: chance of farkling by dice count
    FARKLE_RISK = {
        1: 0.667,
        2: 0.444,
        3: 0.278,
        4: 0.157,
        5: 0.077,
        6: 0.031,
    }

    # Baseline banking thresholds (EV-optimal once on board)
    # Bank if current_score >= threshold for next roll dice count
    BASELINE_THRESHOLDS = {
        1: 250,
        2: 250,
        3: 400,
        4: 1000,
        5: 2900,
        6: 10050,  # basically never
    }

    # Endgame: chance to win this turn based on points needed
    # Format: (points_needed, win_probability)
    ENDGAME_WIN_CHANCES = [
        (300, 0.83),
        (400, 0.67),
        (500, 0.52),
        (600, 0.45),
        (700, 0.38),
        (800, 0.31),
        (900, 0.27),
        (1000, 0.25),
        (1200, 0.15),
        (1500, 0.095),
        (2000, 0.038),
    ]

    def __init__(self, risk_preference: str = "baseline") -> None:
        """Initialize strategy with risk preference.

        Args:
            risk_preference: "conservative", "baseline", or "aggressive"
        """
        self.risk_preference = risk_preference
        self.thresholds = self._compute_thresholds(risk_preference)

    def _compute_thresholds(self, risk_preference: str) -> dict[int, int]:
        """Compute banking thresholds based on risk preference."""
        if risk_preference == "conservative":
            # Bank 100-200 earlier
            return {
                1: 150,
                2: 200,
                3: 300,
                4: 800,
                5: 2700,
                6: 10000,
            }
        elif risk_preference == "aggressive":
            # Bank 100-300 later
            return {
                1: 300,
                2: 300,
                3: 500,
                4: 1200,
                5: 3200,
                6: 10050,
            }
        else:  # baseline
            return self.BASELINE_THRESHOLDS.copy()

    def _get_win_chance(self, points_needed: int) -> float:
        """Get probability of winning this turn given points needed."""
        if points_needed <= 0:
            return 1.0

        # Find closest match in table
        for needed, chance in self.ENDGAME_WIN_CHANCES:
            if points_needed <= needed:
                return chance

        # Beyond table, very low chance
        return 0.01

    def should_roll(
        self,
        turn_num: int,
        total_score: int,
        remaining_dice: int,
        current_score: int,
        winning_score: int = 10000,
        opponent_scores: list[int] | None = None,
    ) -> bool:
        """Decide whether to roll or bank.

        Args:
            turn_num: Current turn number (unused but part of interface)
            total_score: Player's total banked score
            remaining_dice: Number of dice available to roll next
            current_score: Unbanked turn score
            winning_score: Target score to win (default 10000)
            opponent_scores: List of opponent scores for endgame decisions
        """
        # If we can win by banking, always bank
        if total_score + current_score >= winning_score:
            return False

        # Calculate points needed to win
        points_needed = winning_score - total_score

        # Endgame racing logic (no last turn)
        if opponent_scores:
            closest_opponent = max(opponent_scores) if opponent_scores else 0
            opponent_needs = winning_score - closest_opponent

            # If any opponent is close (within 600 of winning)
            if opponent_needs <= 600:
                # They're 45-67% to win on their next turn
                our_points_to_go = points_needed - current_score

                # If we can win this turn with reasonable chance, go for it
                if our_points_to_go <= 600:
                    # We have similar chance, race for it
                    threshold = our_points_to_go
                else:
                    # We're too far behind, this is hail mary territory
                    # Use aggressive thresholds
                    threshold = self.thresholds[remaining_dice] + 200
            else:
                # Everyone is far, use baseline thresholds
                threshold = self.thresholds[remaining_dice]
        else:
            # No opponent info, use baseline thresholds
            threshold = self.thresholds[remaining_dice]

            # In late game with no opponent info, be more aggressive
            if points_needed <= 1000:
                threshold = min(threshold, points_needed)

        # Don't roll if we've hit threshold
        if current_score >= threshold:
            return False

        # Opening rule: if not on board yet, we must bank at least 1000
        # But if current implementation uses different MIN_FIRST_BANK, respect that
        # The strategy caller should enforce MIN_FIRST_BANK separately

        return True

    def actions(
        self,
        die_rolls: list[int],
        depth: int = 0,
    ) -> list[str]:
        """Choose which dice to take.

        Tactical priority:
        1. Triples (highest value first)
        2. Ones (100 each)
        3. Fives (50 each) - but prefer keeping dice count high early in turn
        4. Two hundred combo (three 2s) - lowest priority

        Args:
            die_rolls: List of dice values
            depth: Recursion depth for handling multiple scoring groups
        """
        actions: list[str] = []
        scorer = Scorer(die_rolls)
        remaining_dice = die_rolls

        if scorer.is_blown():
            return actions

        # Priority 1: Take highest value triples first
        if scorer.has_thousand_combo():  # Three 1s = 1000
            actions.append("take_thousand_combo")
            _add_score, remaining_dice = scorer.take_thousand_combo()
        elif scorer.has_six_hundred_combo():  # Three 6s = 600
            actions.append("take_six_hundred_combo")
            _add_score, remaining_dice = scorer.take_six_hundred_combo()
        elif scorer.has_five_hundred_combo():  # Three 5s = 500
            actions.append("take_five_hundred_combo")
            _add_score, remaining_dice = scorer.take_five_hundred_combo()
        elif scorer.has_four_hundred_combo():  # Three 4s = 400
            actions.append("take_four_hundred_combo")
            _add_score, remaining_dice = scorer.take_four_hundred_combo()
        elif scorer.has_three_hundred_combo():  # Three 3s = 300
            actions.append("take_three_hundred_combo")
            _add_score, remaining_dice = scorer.take_three_hundred_combo()
        elif scorer.has_two_hundred_combo():  # Three 2s = 200
            actions.append("take_two_hundred_combo")
            _add_score, remaining_dice = scorer.take_two_hundred_combo()
        # Priority 2: Take ones (better than fives for value and dice management)
        elif scorer.has_ones():
            # At 6 dice early in turn, sometimes take just one 1 to keep dice count high
            num_ones = scorer.num_ones()
            if depth == 0 and len(die_rolls) == 6 and num_ones == 1:
                # Take just one to maintain dice advantage
                actions.append("take_one")
                _add_score, remaining_dice = scorer.take_ones(1)
            else:
                # Take all ones
                for _num in range(num_ones):
                    actions.append("take_one")
                _add_score, remaining_dice = scorer.take_ones(num_ones)
        # Priority 3: Take fives (lowest scoring option)
        elif scorer.has_fifties():
            num_fifties = scorer.num_fifties()
            # Early in turn at 6 dice, consider taking just one 5 to keep dice count
            if depth == 0 and len(die_rolls) == 6 and num_fifties >= 2:
                # Take just one five to maintain dice advantage
                actions.append("take_fifty")
                _add_score, remaining_dice = scorer.take_fifties(1)
            else:
                # Take all fifties
                for _num in range(num_fifties):
                    actions.append("take_fifty")
                _add_score, remaining_dice = scorer.take_fifties(num_fifties)
        else:
            # Should not happen if not blown
            raise AssertionError(f"No valid actions but not blown: {die_rolls}")

        # Recursively process remaining dice
        new_actions: list[str] = []
        if actions and remaining_dice:
            new_actions = self.actions(remaining_dice, depth + 1)

        return actions + new_actions
