import random
from typing import Protocol

from scorer import Scorer


class Strategy(Protocol):
    def should_roll(
        self, turn_num: int, total_score: int, remaining_dice: int, current_score: int
    ) -> bool: ...

    def actions(self, die_rolls: list[int]) -> list[str]: ...


TurnAction = str | tuple[str, list[int]] | tuple[str, int, int]


class YouBlewIt:
    strategy: Strategy
    num_turns: int | None
    stop_score: int | None
    total_score: int

    def __init__(
        self,
        strategy: Strategy,
        num_turns: int | None = None,
        stop_score: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.num_turns = num_turns
        self.stop_score = stop_score
        self.total_score = 0

    def play(self) -> tuple[int, int]:
        if self.num_turns:
            self._play_turns()
        else:
            self.num_turns = self._play_score()
        assert self.num_turns is not None
        return (self.total_score, self.num_turns)

    def _play_score(self) -> int:
        turn_num = 0
        assert self.stop_score is not None
        while self.total_score < self.stop_score:
            score, _actions = self._play_turn(turn_num)
            self.total_score = self.total_score + score
            turn_num = turn_num + 1
        return turn_num

    def _play_turns(self) -> None:
        assert self.num_turns is not None
        for turn_num in range(self.num_turns):
            score, _actions = self._play_turn(turn_num)
            self.total_score = self.total_score + score

    def _play_turn(self, turn_num: int, current_score: int = 0) -> tuple[int, list[TurnAction]]:
        num_remaining_dice = 6
        turn_actions: list[TurnAction] = []
        new_actions: list[TurnAction] = []
        scorer = Scorer([])
        while self._should_roll(turn_num, num_remaining_dice, current_score):
            die_rolls = self._roll(num_remaining_dice)
            turn_actions.append(("rolled", die_rolls))
            scorer = Scorer(die_rolls)
            if scorer.is_blown():
                return 0, turn_actions + ["blew it"]
            actions = self.strategy.actions(die_rolls)
            turn_actions += actions
            score = scorer.apply_actions(actions)
            current_score = current_score + score
            turn_actions.append(("adding", score, current_score))
            num_remaining_dice = scorer.num_remaining_dice()
            num_remaining_dice = num_remaining_dice if num_remaining_dice != 0 else 6
        dice = scorer._make_remaining_dice()
        num_remaining, raw_score = Scorer(dice).raw_score()
        current_score += raw_score
        turn_actions.append(("auto-adding", raw_score, current_score))
        game_over = self.stop_score and current_score + self.total_score >= self.stop_score
        if num_remaining == 0 and not game_over:
            turn_actions.append("rolled over")
            current_score, new_actions = self._play_turn(turn_num, current_score)
        return (current_score, turn_actions + new_actions)

    def _should_roll(self, turn_num: int, remaining_dice: int, current_score: int) -> bool:
        has_to = self.total_score == 0 and current_score < 1000
        game_over = self.stop_score and current_score + self.total_score >= self.stop_score
        strategy_says = self.strategy.should_roll(
            turn_num, self.total_score, remaining_dice, current_score
        )
        if has_to:
            return True
        if game_over:
            return False
        return strategy_says

    def _roll(self, remaining_dice: int) -> list[int]:
        return [random.randint(1, 6) for _ in range(remaining_dice)]
