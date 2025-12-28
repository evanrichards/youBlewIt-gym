import gymnasium as gym

import gym_env.you_blew_it  # noqa: F401
import gym_env.you_blew_it_v2  # noqa: F401

gym.register(
    id="YouBlewIt-v2", entry_point="gym_env.you_blew_it_v2:YouBlewItV2Env", max_episode_steps=1000
)
gym.register(
    id="YouBlewIt-v1", entry_point="gym_env.you_blew_it:YouBlewItEnv", max_episode_steps=1000
)
