import gymnasium as gym

import gym_env.you_blew_it  # noqa: F401
import gym_env.you_blew_it_v2  # noqa: F401
import gym_env.you_blew_it_1v1  # noqa: F401

from strategies.basic_strategy import BasicStrategy
from strategies.moms_strategy import MomsStrategy
from strategies.evans_strategy import EvansStrategy
from strategies.random_strategy import RandomStrategy

gym.register(
    id="YouBlewIt-v2", entry_point="gym_env.you_blew_it_v2:YouBlewItV2Env", max_episode_steps=1000
)
gym.register(
    id="YouBlewIt-v1", entry_point="gym_env.you_blew_it:YouBlewItEnv", max_episode_steps=1000
)

# 1v1 environments with different opponent strategies
gym.register(
    id="YouBlewIt-1v1-basic",
    entry_point="gym_env.you_blew_it_1v1:YouBlewIt1v1Env",
    max_episode_steps=1000,
    kwargs={"opponent_strategy": BasicStrategy()},
)
gym.register(
    id="YouBlewIt-1v1-moms",
    entry_point="gym_env.you_blew_it_1v1:YouBlewIt1v1Env",
    max_episode_steps=1000,
    kwargs={"opponent_strategy": MomsStrategy()},
)
gym.register(
    id="YouBlewIt-1v1-evans",
    entry_point="gym_env.you_blew_it_1v1:YouBlewIt1v1Env",
    max_episode_steps=1000,
    kwargs={"opponent_strategy": EvansStrategy({1: 300, 2: 300, 3: 350, 4: 400, 5: 500, 6: 600})},
)
gym.register(
    id="YouBlewIt-1v1-random",
    entry_point="gym_env.you_blew_it_1v1:YouBlewIt1v1Env",
    max_episode_steps=1000,
    kwargs={"opponent_strategy": RandomStrategy()},
)
