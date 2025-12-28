import os
import time

import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

env_name = "YouBlewIt-v1"

wandb.init(
    sync_tensorboard=True,
    name=env_name + f"_ppo{int(time.time())}",
)

models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
env = Monitor(gym.make(env_name))
pickup_step = 0
TIMESTEPS = 10000
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{env_name}")
# pickup_step = 200000
# model = PPO.load(
#     os.path.join(models_dir, "PPO_" + env_name + "_" + str(pickup_step)),
#     tensorboard_log=f"runs/{env_name}",
#     env=env)
iters = 0
while iters < 10:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=WandbCallback())
    model_path = os.path.join(
        models_dir, "PPO_" + env_name + "_" + str(iters * TIMESTEPS + pickup_step)
    )
    print(f"iter {iters}: Saving model to {model_path}")
    model.save(model_path)

obs, _ = env.reset()
for _i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
actions_list = [" stop", "1s", "2s", "3s", "4s", "5s", "6s", "50", "100", "roll"]
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    print(f"given obs of {obs}, action is {actions_list[action]}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward of {reward} and done is {terminated or truncated} and info is {info}")
    if terminated or truncated:
        print("new game")
        obs, _ = env.reset()
    # wait for user input
    input()
env.close()
