import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class YouBlewItV2Env(gym.Env):
    # action space consists of one of every combo (6), one and five (2), roll and stop (2)
    # stop, 1s, 2s, 3s, 4s, 5s, 6s, 50, 100, roll
    action_space = spaces.Discrete(10)

    # observation space consists of number of die left by index, has combos:
    # 1dl 2dl 3dl 4dl 5dl 6dl 1000 200 300 400 500 600 50 100, needs roll
    # [ 0,0,0,0,1,0,0,0,0,0,0,0,0,1] # 5 die left, 100 on the board
    observation_space = spaces.Discrete(15)

    def __init__(self):
        self.must_roll = False
        self.blow = False
        self.score = 0
        self.max_score = 10000
        self.just_rolled = False
        self.unbanked_score = 0
        self.dice = [0,0,0,0,0,0]
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
            return self._get_observation(), 0, False, {}
        self.just_rolled = False
        if self.must_roll:
            return self._illegal_move("in must roll state")            
        if action == 0:
            self.score += self.unbanked_score
            self.unbanked_score = 0
            self.must_roll = True
            return self._get_observation(), 0, self.score >= self.max_score, {}
        if action >= 1 and action <= 6:
            if not self._has_num_dice(action):
                return self._illegal_move("tried to take a combo that was not there")
            self._remove_dice(action, 3)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), score_for_action(action), False, {}
        if action == 7 or action == 8:
            number = 5 if action == 7 else 1
            if not self._has_num_dice(number, 1):
                return self._illegal_move("tried to take a die that was not there")
            self._remove_dice(number, 1)
            self.unbanked_score += score_for_action(action)
            return self._get_observation(), score_for_action(action), False, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
    # 1dl 2dl 3dl 4dl 5dl 6dl 1000 200 300 400 500 600 50 100 needs roll
    # [ 0,0,0,0,1,0,0,0,0,0,0,0,0,1] # 5 die left, 100 on the board
        self.just_rolled = False
        self.must_roll = True
        self.score = 0
        self.blown = False
        self.unbanked_score = 0
        self.dice = [0,0,0,0,0,0]
        state = np.zeros(15, dtype=int)
        state[14] = 1
        return state

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
        return self.reset(), -1, True, {"reason": reason}    

    def _has_num_dice(self, number, num_dice=3):
        return len(filter(lambda x: x == number, self.dice)) >= num_dice

    def _remove_dice(self, die_number, number_of_die):
        index = 0
        while number_of_die != 0:
            if self.dice[index] == die_number:
                self.dice[index] = 0
                number_of_die -= 1
            index += 1
        self.must_roll = sum(self.dice) == 0

    def _get_observation(self):
    # 0     1   2   3   4   5   6   7   8   9   10  11  12  13  14
    # 1dl 2dl 3dl 4dl 5dl 6dl 1000 200 300 400 500 600 50 100  needs roll
    # [ 0,0,0,0,1,0,0,0,0,0,0,0,0,1] # 5 die left, 100 on the board
        state = np.zeros(15, dtype=int)
        state[self.num_remaining_dice-1] = 1
        actions = self.legal_actions
        if actions == [9]:
            state[14] = 1
            return state
        for action in actions:
            state[action + 5] = 1
        state[14] = 0
        return state

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
        for i in xrange(6):
            if self.dice[i] != 0:
                self.dice[i] = self.np_random.randint(1, 7)

    def _roll_all(self):
        self.dice = list(self.np_random.randint(1, 7, 6))

    @property
    def legal_actions(self):
        if self.blown or self.must_roll:
            return [9]
        actions = []
        for i in xrange(1,7):
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
    def num_remaining_dice(self):
        return len(filter(lambda x: x != 0, self.dice))

def score_for_action(action):
    if action == 0:
        return 1
    if action == 1:
        return 1000
    if action <=6:
        return action * 100
    if action == 7:
        return 50
    if action == 8:
        return 100
    return 0

