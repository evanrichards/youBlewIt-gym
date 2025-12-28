import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

StepResult = tuple[NDArray[np.int64], float, bool, bool, dict[str, str]]

# Minimum score needed to "get on the board" with your first bank
MIN_FIRST_BANK = 1000


class YouBlewItEnv(gym.Env[NDArray[np.int64], int]):
    # action space consists of one of every combo (6), one and five (2), roll and stop (2)
    # stop, 1s, 2s, 3s, 4s, 5s, 6s, 50, 100, roll
    action_space: spaces.Discrete = spaces.Discrete(10)
    observation_space: spaces.MultiDiscrete

    must_roll: bool
    blow: bool
    score: int
    max_score: int
    just_rolled: bool
    unbanked_score: int
    dice: list[int]
    blown: bool

    def __init__(self) -> None:
        self.must_roll = False
        self.blow = False
        self.score = 0
        self.max_score = 10000
        self.just_rolled = False
        self.unbanked_score = 0
        self.dice = [0, 0, 0, 0, 0, 0]
        self.observation_space = spaces.MultiDiscrete([7, 7, 7, 7, 7, 7, self.max_score + 1, 2])
        self.blown = False

    def step(self, action: int) -> StepResult:
        """Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (observation, reward, terminated, truncated, info).
        """
        if not self.action_space.contains(action):
            return self._illegal_move("no such action")
        if action == 9:
            if self.just_rolled:
                return self._illegal_move("rolled twice in a row without blowing it")
            self.just_rolled = True
            self._roll()
            return self._get_observation(), -10.0, False, False, {}
        self.just_rolled = False
        if self.must_roll:
            return self._illegal_move("in must roll state")
        if action == 0:
            self.score += self.unbanked_score
            reward = float(self.unbanked_score)
            self.unbanked_score = 0
            self.must_roll = True
            return self._get_observation(), reward, self.score >= self.max_score, False, {}
        if action >= 1 and action <= 6:
            if not self._has_num_dice(action):
                return self._illegal_move("tried to take a combo that was not there")
            self._remove_dice(action, 3)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), self.reward(), False, False, {}
        if action == 7 or action == 8:
            number = 5 if action == 7 else 1
            if not self._has_num_dice(number, 1):
                return self._illegal_move("tried to take a die that was not there")
            self._remove_dice(number, 1)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), self.reward(), False, False, {}
        return self._illegal_move("unknown action")

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.int64], dict[str, object]]:
        """Resets the state of the environment and returns an initial observation."""
        super().reset(seed=seed)
        self.just_rolled = False
        self.must_roll = True
        self.score = 0
        self.blown = False
        self.unbanked_score = 0
        self.dice = [0, 0, 0, 0, 0, 0]
        return np.array(self.dice + [0, 0], dtype=np.int64), {}

    def _illegal_move(self, reason: str) -> StepResult:
        obs, _ = self.reset()
        return obs, -10.0, True, False, {"reason": reason}

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
        return np.array(self.dice + [self.unbanked_score, int(self.blown)], dtype=np.int64)

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

    def num_remaining_dice(self) -> int:
        return len([x for x in self.dice if x != 0])

    def reward(self) -> float:
        return (float(self.unbanked_score) / float(self.max_score)) * 50.0


def score_for_action(action: int) -> int:
    if action == 1:
        return 1000
    if action <= 6:
        return action * 100
    if action == 7:
        return 50
    if action == 8:
        return 100
    return 0
