import json

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

import gym_env  # noqa: F401
from gym_env.you_blew_it_v2 import YouBlewItV2Env, score_for_action

env = gym.make("YouBlewIt-v2")


def random_legal_moves(env: gym.Env[NDArray[np.int64], int], num_episodes: int = 100) -> float:
    all_steps: list[int] = []
    unwrapped: YouBlewItV2Env = env.unwrapped  # type: ignore[assignment]
    for _g in range(num_episodes):
        _s, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = int(np.random.choice(unwrapped.legal_actions))
            new_s, _r, terminated, truncated, log = env.step(action)
            done = terminated or truncated
            if action == 0 or not new_s[7]:
                steps += 1
            if log:
                print(log)
        all_steps.append(steps)
    return float(np.average(all_steps))


def greedy_treshold(
    env: gym.Env[NDArray[np.int64], int], threshold: float = 0.3, num_episodes: int = 100
) -> float:
    # given the number of dice i have left what is the percentage of blown times on rolls
    r_table = np.zeros((6, 2))
    unwrapped: YouBlewItV2Env = env.unwrapped  # type: ignore[assignment]

    all_steps: list[int] = []
    eps = 0.5
    decay_factor = 0.999
    for _g in range(num_episodes):
        _s, _ = env.reset()
        done = False
        steps = 0
        eps *= decay_factor
        remaining_count = 0
        while not done:
            actions = np.array(unwrapped.legal_actions)
            point_actions = np.setdiff1d(actions, [0, 9])
            if len(point_actions) != 0:
                action = int(max(actions, key=score_for_action))
            else:
                remaining_count = unwrapped.num_remaining_dice - 1
                blown_percentage = (
                    0.0
                    if not r_table[remaining_count][1]
                    else r_table[remaining_count][0] / r_table[remaining_count][1]
                )
                if 0 not in actions or np.random.random() < eps or blown_percentage < threshold:
                    action = 9
                    r_table[remaining_count][1] += 1
                else:
                    action = 0
            new_s, _r, terminated, truncated, log = env.step(action)
            done = terminated or truncated
            safe = new_s[7]
            if action == 9 and not safe:
                steps += 1
                r_table[remaining_count][0] += 1
            elif action == 0:
                steps += 1
            if log:
                print(log)
        all_steps.append(steps)
    return float(np.average(all_steps))


def predictive_potential(env: gym.Env[NDArray[np.int64], int], num_episodes: int = 500) -> float:
    # given the number of dice i have left what is the average earning [count, total]
    r_table = np.zeros((6, 2))
    lr = 0.8
    y = 0.95
    unwrapped: YouBlewItV2Env = env.unwrapped  # type: ignore[assignment]

    all_steps: list[int] = []
    eps = 0.5
    decay_factor = 0.999
    for _g in range(num_episodes):
        s, _ = env.reset()
        done = False
        steps = 0
        eps *= decay_factor
        pre_banked_points: NDArray[np.float64] = np.array([])
        num_die_left_list: list[int] = []
        turn_logs: list[list[str]] = []
        average_return = 0.0
        num_die_left = 0
        while not done:
            actions = np.array(unwrapped.legal_actions)
            num_die_left = unwrapped.num_remaining_dice
            point_actions = np.setdiff1d(actions, [0, 9])
            turn_logs.append(
                [
                    "prebanked: " + str(pre_banked_points),
                    "die left: " + str(num_die_left),
                    "available actions: " + str(actions),
                ]
            )
            if np.random.random() < eps:
                action = int(np.random.choice(actions))
            elif len(actions) == 1:
                turn_logs[-1].append("choice -z")
                action = int(actions[0])
            elif 9 in actions:
                turn_logs[-1].append("choice a")
                # here we have an option to roll or take more
                count, total = r_table[num_die_left - 1]
                max_action = int(max(actions, key=score_for_action))
                average_return = total / count if count != 0 else num_die_left * 100.0
                turn_logs[-1].append(f"average_return: {average_return}, max_action: {max_action}")
                action = max_action if score_for_action(max_action) > average_return else 9
            elif 0 in actions and len(point_actions) == 0:
                turn_logs[-1].append("choice b")
                action = 0
            else:
                turn_logs[-1].append("choice c")
                action = int(max(actions, key=score_for_action))
            turn_logs[-1].append(f"chosen action: {action}")
            new_s, r, terminated, truncated, log = env.step(action)
            done = terminated or truncated
            blown = new_s[7]
            turn_logs[-1].append(f"old state: {s}")
            turn_logs[-1].append(f"new state: {new_s}")
            s = new_s

            if r != 0:
                pre_banked_points = np.append(pre_banked_points, [0])

                num_die_left = unwrapped.num_remaining_dice
                count, total = r_table[num_die_left - 1]
                new_average_return = total / count if count != 0 else num_die_left * 100.0

                pre_banked_points = np.add(
                    pre_banked_points,
                    [r + lr * (y * new_average_return - average_return)],
                )
                num_die_left_list.append(num_die_left)
            if action == 9 and blown:
                turn_logs = []
                # blew it case
                steps += 1
                if num_die_left_list:
                    r_table[num_die_left - 1][0] += 1
                    r_table[num_die_left - 1][1] -= float(np.average(pre_banked_points))
                num_die_left_list = []
                pre_banked_points = np.array([])
            elif action == 0:
                turn_logs = []
                for i, count_val in enumerate(num_die_left_list):
                    r_table[count_val - 1][0] += 1
                    r_table[count_val - 1][1] += pre_banked_points[i]
                steps += 1
                num_die_left_list = []
                pre_banked_points = np.array([])
            if log:
                print(json.dumps(turn_logs))
                print(log)
        all_steps.append(steps)
    for row in r_table:
        print(row, row[1] / row[0])
    return float(np.average(all_steps))


def learn_from_scratch(env: gym.Env[NDArray[np.int64], int], num_episodes: int = 5000000) -> float:
    # given the number of dice i have left what is the average earning [count, total]
    assert hasattr(env.observation_space, "n") and hasattr(env.action_space, "n")
    obs_n: int = env.observation_space.n  # type: ignore[attr-defined]
    act_n: int = env.action_space.n  # type: ignore[attr-defined]
    r_table = np.zeros((obs_n, act_n))
    lr = 0.8
    discount_rate = 0.99
    action_dict: dict[int, int] = {}
    all_steps = np.zeros(10, dtype=int)
    eps = 0.9
    decay_factor = 0.999
    games_failed = 0
    for g in range(num_episodes):
        if g % 10000 == 0:
            print(g)
        s, _ = env.reset()
        done = False
        steps = 0
        eps *= decay_factor
        while not done:
            if np.random.random() < eps:
                action = int(env.action_space.sample())
            else:
                action = int(np.argmax(r_table[s, :]))
            action_dict[action] = action_dict.get(action, 0) + 1
            new_s, reward, terminated, truncated, log = env.step(action)
            done = terminated or truncated
            updated = r_table[s, action] * (1 - lr) + lr * (
                float(reward) + discount_rate * np.max(r_table[new_s, :])
            )
            r_table[s, action] = updated
            s = new_s
            steps += 1
            if log:
                games_failed += 1
        all_steps[g % 10] = steps
    print(games_failed / num_episodes)
    print(action_dict)
    print(r_table)
    print(all_steps)
    return float(np.average(all_steps))


if __name__ == "__main__":
    env.reset(seed=1)
    print(learn_from_scratch(env))
