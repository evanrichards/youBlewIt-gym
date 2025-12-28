from collections.abc import Sequence
from statistics import mean

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from game import YouBlewIt
from strategies import EvansStrategy

x0 = np.array([0.0, 200.0, 200.0, 200.0, 2000.0, 2000.0])


def f(x: Sequence[float]) -> float:
    one, two, three, four, five, six = x
    num_games = 1000
    turns_list: list[int] = []
    basic_strategy = EvansStrategy(
        {
            6: int(six),
            5: int(five),
            4: int(four),
            3: int(three),
            2: int(two),
            1: int(one),
        }
    )
    for _num in range(num_games):
        ybi = YouBlewIt(basic_strategy, stop_score=10000)
        _score, turns = ybi.play()
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


def cb(xb: NDArray[np.float64]) -> None:
    print(xb)


result = minimize(f, x0, method="Nelder-Mead", callback=cb, options={"disp": True})
print(result)
