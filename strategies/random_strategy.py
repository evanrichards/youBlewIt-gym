import random

from scorer import Scorer


class RandomStrategy:
    def __init__(self) -> None:
        pass

    def should_roll(
        self, turn_num: int, total_score: int, remaining_dice: int, current_score: int
    ) -> bool:
        return random.choice([True, False])

    def actions(self, die_rolls: list[int]) -> list[str]:
        """Take random legal actions until at least one scoring die is taken."""
        actions: list[str] = []
        scorer = Scorer(die_rolls)

        if scorer.is_blown():
            return []

        # Keep taking random actions until we've taken at least one
        while True:
            available = self._get_available_actions(scorer)
            if not available:
                break

            # Pick a random action
            action = random.choice(available)
            actions.append(action)

            # Execute it on the scorer to update state
            method = getattr(scorer, action)
            method()

            # Randomly decide whether to keep taking more
            if actions and random.choice([True, False]):
                break

        return actions

    def _get_available_actions(self, scorer: Scorer) -> list[str]:
        """Get list of currently available actions given scorer state."""
        available: list[str] = []
        # Check values directly since combos isn't updated after takes
        v = scorer.values
        if v[1] is not None and v[1] >= 3:
            available.append("take_thousand_combo")
        if v[6] is not None and v[6] >= 3:
            available.append("take_six_hundred_combo")
        if v[5] is not None and v[5] >= 3:
            available.append("take_five_hundred_combo")
        if v[4] is not None and v[4] >= 3:
            available.append("take_four_hundred_combo")
        if v[3] is not None and v[3] >= 3:
            available.append("take_three_hundred_combo")
        if v[2] is not None and v[2] >= 3:
            available.append("take_two_hundred_combo")
        if v[1] is not None and v[1] >= 1:
            available.append("take_one")
        if v[5] is not None and v[5] >= 1:
            available.append("take_fifty")
        return available
