from scorer import Scorer


class BasicStrategy:
    def __init__(self):
        pass

    def should_roll(self, turn_num, total_score, remaining_dice, current_score):
        return not remaining_dice <= 2

    def actions(self, die_rolls):
        # print "scoring", die_rolls
        actions = []
        scorer = Scorer(die_rolls)
        remaining_dice = die_rolls
        if scorer.is_blown():
            pass
        elif scorer.has_thousand_combo():
            actions.append("take_thousand_combo")
            add_score, remaining_dice = scorer.take_thousand_combo()
        elif scorer.has_six_hundred_combo():
            actions.append("take_six_hundred_combo")
            add_score, remaining_dice = scorer.take_six_hundred_combo()
        elif scorer.has_five_hundred_combo():
            actions.append("take_five_hundred_combo")
            add_score, remaining_dice = scorer.take_five_hundred_combo()
        elif scorer.has_four_hundred_combo():
            actions.append("take_four_hundred_combo")
            add_score, remaining_dice = scorer.take_four_hundred_combo()
        elif scorer.has_three_hundred_combo():
            actions.append("take_three_hundred_combo")
            add_score, remaining_dice = scorer.take_three_hundred_combo()
        elif scorer.has_ones():
            num_ones = scorer.num_ones()
            for _num in range(num_ones):
                actions.append("take_one")
            add_score, remaining_dice = scorer.take_ones(num_ones)
        elif scorer.has_two_hundred_combo():
            add_score, remaining_dice = scorer.take_two_hundred_combo()
        elif scorer.has_fifties():
            num_fifties = scorer.num_fifties()
            for _num in range(num_fifties):
                actions.append("take_fifty")
            add_score, remaining_dice = scorer.take_fifties(num_fifties)
        else:
            raise AssertionError(f"wtf {scorer.is_blown()} {die_rolls}")
        # print "for", add_score, "points"
        new_actions = []
        if actions:
            new_actions = self.actions(remaining_dice)
        return actions + new_actions
