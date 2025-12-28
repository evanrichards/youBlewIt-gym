from typing import Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from scorer import Scorer


class Strategy(Protocol):
    def should_roll(
        self, turn_num: int, total_score: int, remaining_dice: int, current_score: int
    ) -> bool: ...

    def actions(self, die_rolls: list[int]) -> list[str]: ...


StepResult = tuple[NDArray[np.int64], float, bool, bool, dict[str, str]]

# Score buckets: 0-2k, 2k-4k, 4k-6k, 6k-8k, 8k-10k
NUM_SCORE_BUCKETS = 5
SCORE_BUCKET_SIZE = 2000

WIN_BONUS = 1000.0

# Minimum score needed to "get on the board" with your first bank
MIN_FIRST_BANK = 1000


def score_to_bucket(score: int) -> int:
    return min(score // SCORE_BUCKET_SIZE, NUM_SCORE_BUCKETS - 1)


class YouBlewIt1v1Env(gym.Env[NDArray[np.int64], int]):
    """1v1 competitive You Blew It environment.

    The agent plays against an opponent using a fixed strategy.
    Turns alternate: agent acts, then opponent takes a full turn.
    Game ends when either player reaches max_score.

    Observation space (25 dims):
        - Positions 0-5: number of dice remaining (one-hot)
        - Positions 6-13: available scoring actions
        - Position 14: needs roll flag
        - Positions 15-19: agent score bucket (one-hot)
        - Positions 20-24: opponent score bucket (one-hot)

    Rewards:
        - Banking points: amount banked
        - Winning: +WIN_BONUS
        - Illegal move: -1
    """

    action_space: spaces.Discrete = spaces.Discrete(10)
    observation_space: spaces.MultiBinary = spaces.MultiBinary(25)

    must_roll: bool
    score: int
    opponent_score: int
    max_score: int
    just_rolled: bool
    unbanked_score: int
    dice: list[int]
    blown: bool
    opponent_strategy: Strategy
    turn_num: int

    def __init__(self, opponent_strategy: Strategy) -> None:
        self.opponent_strategy = opponent_strategy
        self.must_roll = False
        self.score = 0
        self.opponent_score = 0
        self.max_score = 10000
        self.just_rolled = False
        self.unbanked_score = 0
        self.dice = [0, 0, 0, 0, 0, 0]
        self.blown = False
        self.turn_num = 0

    def step(self, action: int) -> StepResult:
        if not self.action_space.contains(action):
            return self._illegal_move("no such action")

        if action == 9:  # Roll
            if self.just_rolled:
                return self._illegal_move("rolled twice in a row without blowing it")
            self.just_rolled = True
            self._roll()
            return self._get_observation(), 0.0, False, False, {}

        self.just_rolled = False

        if self.must_roll:
            return self._illegal_move("in must roll state")

        if action == 0:  # Stop/bank
            banked = self.unbanked_score
            self.score += self.unbanked_score
            self.unbanked_score = 0
            self.must_roll = True

            # Check if agent won
            if self.score >= self.max_score:
                return self._get_observation(), float(banked) + WIN_BONUS, True, False, {"result": "win"}

            # Opponent takes their turn
            self._opponent_turn()

            # Check if opponent won
            if self.opponent_score >= self.max_score:
                return self._get_observation(), float(banked), True, False, {"result": "loss"}

            self.turn_num += 1
            return self._get_observation(), float(banked), False, False, {}

        if action >= 1 and action <= 6:  # Take triple combo
            if not self._has_num_dice(action):
                return self._illegal_move("tried to take a combo that was not there")
            self._remove_dice(action, 3)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), 0.0, False, False, {}

        if action == 7 or action == 8:  # Take single 5 or 1
            number = 5 if action == 7 else 1
            if not self._has_num_dice(number, 1):
                return self._illegal_move("tried to take a die that was not there")
            self._remove_dice(number, 1)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), 0.0, False, False, {}

        return self._illegal_move("unknown action")

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.int64], dict[str, object]]:
        super().reset(seed=seed)
        self.just_rolled = False
        self.must_roll = True
        self.score = 0
        self.opponent_score = 0
        self.blown = False
        self.unbanked_score = 0
        self.dice = [0, 0, 0, 0, 0, 0]
        self.turn_num = 0
        return self._get_observation(), {}

    def _opponent_turn(self) -> None:
        """Simulate a complete turn for the opponent using their strategy."""
        turn_score = self._play_opponent_turn(current_score=0, num_dice=6)
        self.opponent_score += turn_score

    def _play_opponent_turn(self, current_score: int, num_dice: int) -> int:
        """Recursively play opponent's turn, handling hot dice (rolling all 6 again)."""
        while self._opponent_should_roll(num_dice, current_score):
            dice = self._roll_n_dice(num_dice)
            scorer = Scorer(dice)

            if scorer.is_blown():
                return 0  # Lost all unbanked points

            actions = self.opponent_strategy.actions(dice)
            turn_score = scorer.apply_actions(actions)
            current_score += turn_score
            num_dice = scorer.num_remaining_dice()

            # Hot dice: used all dice, roll all 6 again
            if num_dice == 0:
                num_dice = 6

        return current_score

    def _opponent_should_roll(self, remaining_dice: int, current_score: int) -> bool:
        """Check if opponent wants to roll based on their strategy."""
        # Must roll if not yet on board and haven't reached minimum
        if self.opponent_score == 0 and current_score < MIN_FIRST_BANK:
            return True
        return self.opponent_strategy.should_roll(
            self.turn_num, self.opponent_score, remaining_dice, current_score
        )

    def _roll_n_dice(self, n: int) -> list[int]:
        """Roll n dice and return the values."""
        return [int(x) for x in self.np_random.integers(1, 7, n)]

    def _illegal_move(self, reason: str) -> StepResult:
        obs, _ = self.reset()
        return obs, -1.0, True, False, {"reason": reason}

    def _has_num_dice(self, number: int, num_dice: int = 3) -> bool:
        return len([x for x in self.dice if x == number]) >= num_dice

    def _remove_dice(self, die_number: int, number_of_die: int) -> None:
        index = 0
        while number_of_die != 0:
            if self.dice[index] == die_number:
                self.dice[index] = 0
                number_of_die -= 1
            index += 1
        self.must_roll = sum(self.dice) == 0

    def _get_observation(self) -> NDArray[np.int64]:
        # Positions 0-14: same as v2
        # Positions 15-19: agent score bucket (one-hot)
        # Positions 20-24: opponent score bucket (one-hot)
        state = np.zeros(25, dtype=np.int64)

        # Dice remaining (positions 0-5)
        remaining = self.num_remaining_dice
        if remaining > 0:
            state[remaining - 1] = 1

        # Legal actions (positions 6-14)
        actions = self.legal_actions
        if actions == [9]:
            state[14] = 1
        else:
            for action in actions:
                state[action + 5] = 1
            state[14] = 0

        # Score buckets
        state[15 + score_to_bucket(self.score)] = 1
        state[20 + score_to_bucket(self.opponent_score)] = 1

        return state

    def _is_blown(self) -> bool:
        if self.must_roll:
            return False
        count: dict[int, int] = {}
        for die in self.dice:
            count[die] = count.get(die, 0) + 1
        if count.get(1, 0) > 0 or count.get(5, 0) > 0:
            return False
        if count.get(0):
            del count[0]
        if not count:
            return False
        return max(count.values()) < 3

    def _roll(self) -> None:
        if not self.must_roll:
            self._roll_remaining()
        else:
            self._roll_all()
        self.must_roll = sum(self.dice) == 0
        self.blown = self._is_blown()
        if self.blown:
            self._reset_after_blew_it()

    def _reset_after_blew_it(self) -> None:
        self.unbanked_score = 0
        self.must_roll = True
        self.just_rolled = False

    def _roll_remaining(self) -> None:
        for i in range(6):
            if self.dice[i] != 0:
                self.dice[i] = int(self.np_random.integers(1, 7))

    def _roll_all(self) -> None:
        self.dice = [int(x) for x in self.np_random.integers(1, 7, 6)]

    @property
    def legal_actions(self) -> list[int]:
        if self.blown or self.must_roll:
            return [9]
        actions: list[int] = []
        # Can only bank if already on board, or if unbanked meets minimum
        can_bank = self.score > 0 or self.unbanked_score >= MIN_FIRST_BANK
        if self.unbanked_score > 0 and can_bank:
            actions.append(0)
        for i in range(1, 7):
            if self._has_num_dice(i):
                actions.append(i)
        if self._has_num_dice(5, 1):
            actions.append(7)
        if self._has_num_dice(1, 1):
            actions.append(8)
        if not self.just_rolled:
            actions.append(9)
        return actions

    @property
    def num_remaining_dice(self) -> int:
        return len([x for x in self.dice if x != 0])


def score_for_action(action: int) -> int:
    if action == 0:
        return 1
    if action == 1:
        return 1000
    if action <= 6:
        return action * 100
    if action == 7:
        return 50
    if action == 8:
        return 100
    return 0
