import gymnasium as gym

env = gym.make("YouBlewIt-v2")

actions = [9, 8, 9, 7, 9, 7, 9, 8, 9]
env.reset(seed=1)
for action in actions:
    print(env.step(action))
    print(env.unwrapped.legal_actions)
    print(env.unwrapped.dice)
