from scorer import Scorer


class EvansStrategy:
    dice_tolerance: dict[int, int]

    def __init__(self, dice_tolerance: dict[int, int]) -> None:
        self.dice_tolerance = dice_tolerance

    def should_roll(
        self, turn_num: int, total_score: int, remaining_dice: int, current_score: int
    ) -> bool:
        return not current_score >= self.dice_tolerance[remaining_dice]

    def actions(
        self,
        die_rolls: list[int],
        depth: int = 0,
        fifty_taken: bool = False,
        one_taken: bool = False,
    ) -> list[str]:
        actions: list[str] = []
        scorer = Scorer(die_rolls)
        remaining_dice = die_rolls
        if scorer.is_blown():
            pass
        elif scorer.has_thousand_combo():
            actions.append("take_thousand_combo")
            _add_score, remaining_dice = scorer.take_thousand_combo()
        elif scorer.has_six_hundred_combo():
            actions.append("take_six_hundred_combo")
            _add_score, remaining_dice = scorer.take_six_hundred_combo()
        elif scorer.has_five_hundred_combo():
            actions.append("take_five_hundred_combo")
            _add_score, remaining_dice = scorer.take_five_hundred_combo()
        elif scorer.has_four_hundred_combo():
            actions.append("take_four_hundred_combo")
            _add_score, remaining_dice = scorer.take_four_hundred_combo()
        elif scorer.has_three_hundred_combo():
            actions.append("take_three_hundred_combo")
            _add_score, remaining_dice = scorer.take_three_hundred_combo()
        elif depth == 0 and len(die_rolls) == 6 and scorer.has_ones():
            actions.append("take_one")
            _add_score, remaining_dice = scorer.take_ones(1)
            one_taken = True
        elif scorer.has_ones() and not one_taken:
            num_ones = scorer.num_ones()
            for _num in range(num_ones):
                actions.append("take_one")
            _add_score, remaining_dice = scorer.take_ones(num_ones)
        elif depth == 0 and not fifty_taken and scorer.has_fifties():
            actions.append("take_fifty")
            _add_score, remaining_dice = scorer.take_fifties(1)
            fifty_taken = True
        elif scorer.has_two_hundred_combo() and depth == 0:
            _add_score, remaining_dice = scorer.take_two_hundred_combo()
        new_actions: list[str] = []
        if actions:
            new_actions = self.actions(remaining_dice, depth + 1, fifty_taken, one_taken)
        return actions + new_actions
