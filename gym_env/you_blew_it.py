import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class YouBlewItEnv(gym.Env):
    # action space consists of one of every combo (6), one and five (2), roll and stop (2)
    # stop, 1s, 2s, 3s, 4s, 5s, 6s, 50, 100, roll
    action_space = spaces.Discrete(10)

    # observation space consists of all dice positions (6), unbanked score (1), started state (1)

    def __init__(self):
        self.must_roll = False
        self.blow = False
        self.score = 0
        self.max_score = 10000
        self.just_rolled = False
        self.unbanked_score = 0
        self.dice = [0,0,0,0,0,0]
        self.observation_space = spaces.MultiDiscrete([7,7,7,7,7,7,self.max_score + 1, 2])
        self.seed()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if not self.action_space.contains(action):
            return self._illegal_move("no such action")
        if action == 9:
            if self.just_rolled:
                return self._illegal_move("rolled twice in a row without blowing it")
            self.just_rolled = True
            self._roll()
            return self._get_observation(), -10, False, {}
        self.just_rolled = False
        if self.must_roll:
            return self._illegal_move("in must roll state")            
        if action == 0:
            self.score += self.unbanked_score
            reward = self.unbanked_score
            self.unbanked_score = 0
            self.must_roll = True
            return self._get_observation(), reward, self.score >= self.max_score, {}
        if action >= 1 and action <= 6:
            if not self._has_num_dice(action):
                return self._illegal_move("tried to take a combo that was not there")
            self._remove_dice(action, 3)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), self.reward(), False, {}
        if action == 7 or action == 8:
            number = 5 if action == 7 else 1
            if not self._has_num_dice(number, 1):
                return self._illegal_move("tried to take a die that was not there")
            self._remove_dice(number, 1)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), self.reward(), False, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        self.just_rolled = False
        self.must_roll = True
        self.score = 0
        self.blown = False
        self.unbanked_score = 0
        self.dice = [0,0,0,0,0,0]
        return np.array(self.dice + [0, False])

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _illegal_move(self, reason):
        return self.reset(), -10, True, {"reason": reason}    

    def _has_num_dice(self, number, num_dice=3):
        return len(list(filter(lambda x: x == number, self.dice))) >= num_dice

    def _remove_dice(self, die_number, number_of_die):
        index = 0
        while number_of_die != 0:
            if self.dice[index] == die_number:
                self.dice[index] = 0
                number_of_die -= 1
            index += 1
        self.must_roll = sum(self.dice) == 0

    def _get_observation(self):
        return np.array(self.dice + [self.unbanked_score, self.blown])

    def _is_blown(self):
        if self.must_roll:
            return False
        count = {}
        for die in self.dice:
            count[die] = count.get(die, 0) + 1
        if count.get(1,0) > 0 or count.get(5,0) > 0:
            return False
        if count.get(0):
            del count[0]
        return max(count.values()) < 3

    def _roll(self):
        if not self.must_roll:
            self._roll_remaining()
        else:
            self._roll_all()
        self.must_roll = sum(self.dice) == 0
        self.blown = self._is_blown()
        if self.blown:
            self._reset_after_blew_it()

    def _reset_after_blew_it(self):
        self.unbanked_score = 0
        self.must_roll = True
        self.just_rolled = False

    def _roll_remaining(self):
        for i in range(6):
            if self.dice[i] != 0:
                self.dice[i] = self.np_random.randint(1, 7)

    def _roll_all(self):
        self.dice = list(self.np_random.randint(1, 7, 6))

    def legal_actions(self):
        if self.blown or self.must_roll:
            return [9]
        actions = [0,]
        for i in range(1,7):
            if self._has_num_dice(i):
                actions.append(i)
        if self._has_num_dice(5, 1):
            actions.append(7)
        if self._has_num_dice(1, 1):
            actions.append(8)
        if not self.just_rolled:
            actions.append(9)
        return actions

    def num_remaining_dice(self):
        return len(filter(lambda x: x != 0, self.dice))

    def reward(self):
        return (float(self.unbanked_score) / float(self.max_score)) * 50.0


def score_for_action(action):
    if action == 1:
        return 1000
    if action <=6:
        return action * 100
    if action == 7:
        return 50
    if action == 8:
        return 100
    return 0

