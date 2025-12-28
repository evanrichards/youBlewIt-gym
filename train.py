from statistics import mean

import numpy as np
from scipy.optimize import fmin_ncg

from game import YouBlewIt
from strategies import EvansStrategy

# fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None, args=(), iter=100, acc=1e-06, iprint=1, disp=None, full_output=0, epsilon=1.4901161193847656e-08, callback=None)[source]
x0 = np.array([0, 200, 200, 200, 2000, 2000])
ep = np.array([50, 50, 50, 50, 50, 50])


def f(x):
    one, two, three, four, five, six = x
    num_games = 1000
    turns_list = []
    basic_strategy = EvansStrategy(
        {
            6: six,
            5: five,
            4: four,
            3: three,
            2: two,
            1: one,
        }
    )
    for _num in range(num_games):
        ybi = YouBlewIt(basic_strategy, stop_score=10000)
        score, turns = ybi.play()
        turns_list.append(turns)
    return mean(turns_list)


bounds = (
    (0, 10000),
    (0, 10000),
    (0, 10000),
    (0, 10000),
    (0, 10000),
    (0, 10000),
)


def cb(xb):
    print(xb)


print(fmin_ncg(f, x0, epsilon=ep, disp=True, callback=cb))

# print minimize(f, x0,
# 		method='SLSQP',
# 		bounds=bounds,
# 		options={'disp': True, 'eps': 50},
# 		callback=cb)
