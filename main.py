from game import YouBlewIt
from strategies import BasicStrategy, MomsStrategy, EvansStrategy
import numpy as np

total_turns = 0
num_games = 1000
turns_list = []
basic_strategy = EvansStrategy({
			6: 2000,
			5: 2000,
			4: 2000,
			3: 2000,
			2: 2000,
			1: 0,
		})
for num in xrange(num_games):
	ybi = YouBlewIt(basic_strategy, stop_score=10000)
	score, turns = ybi.play()
	turns_list.append(turns)
	total_turns = turns + total_turns

print num_games, np.std(np.array(turns_list)), np.average(np.array(turns_list))
