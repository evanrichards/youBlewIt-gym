from scorer import Scorer


class MomsStrategy:
    def __init__(self) -> None:
        pass

    def should_roll(
        self, turn_num: int, total_score: int, remaining_dice: int, current_score: int
    ) -> bool:
        if current_score >= 1000 and remaining_dice <= 5:
            return False
        if current_score >= 600 and remaining_dice <= 4:
            return False
        if current_score >= 350 and remaining_dice <= 3:
            return False
        if current_score >= 200 and remaining_dice <= 2:
            return False
        return not remaining_dice < 2

    def actions(self, die_rolls: list[int], depth: int = 0, fifty_taken: bool = False) -> list[str]:
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
        elif scorer.has_ones():
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
            new_actions = self.actions(remaining_dice, depth + 1, fifty_taken)
        return actions + new_actions
