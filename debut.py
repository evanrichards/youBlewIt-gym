import gym
import gym_env
env = gym.make("YouBlewIt-v2")

actions = [9, 8, 9, 7, 9, 7, 9, 8, 9]
env.seed(1)
env.reset()
for action in actions:
	print env.step(action)
	print env.legal_actions
	print env.dice