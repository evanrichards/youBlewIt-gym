import gymnasium as gym

import gym_env  # noqa: F401
from gym_env.you_blew_it_v2 import YouBlewItV2Env

env = gym.make("YouBlewIt-v2")
unwrapped_env: YouBlewItV2Env = env.unwrapped  # type: ignore[assignment]

actions = [9, 8, 9, 7, 9, 7, 9, 8, 9]
env.reset(seed=1)
for action in actions:
    print(env.step(action))
    print(unwrapped_env.legal_actions)
    print(unwrapped_env.dice)
