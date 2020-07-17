import gym
import gym_env
import numpy as np
 
# from keras.models import Sequential
# from keras.layers import Dense, InputLayer
env = gym.make("YouBlewIt-v2")

def random_legal_moves(env, num_episodes=100):
    all_steps = []
    for g in range(num_episodes):
        s = env.reset()
        done = False
        steps = 0
        while not done:
            action = np.random.choice(env.legal_actions())
            new_s, r, done, log = env.step(action)
            if action == 0:
                steps += 1
            elif not new_s[7]:
                steps += 1
            if log:
                print(log)
        all_steps.append(steps)
    return np.average(all_steps)

def greedy_treshold(env, threshold=0.3, num_episodes=100):
    # given the number of dice i have left what is the percentage of blown times on rolls
    r_table = np.zeros((6, 2))

    all_steps = []
    eps = 0.5
    decay_factor = 0.999
    for g in range(num_episodes):
        s = env.reset()
        done = False
        steps = 0
        eps *= decay_factor
        while not done:
            actions = np.array(env.legal_actions())
            point_actions = np.setdiff1d(actions, [0,9])
            if len(point_actions) != 0:
                action = max(actions, key=you_blew_it.score_for_action)
            else:
                remaining_count = env.num_remaining_dice() - 1
                blown_percentage = 0 if not r_table[remaining_count][1] else r_table[remaining_count][0] / r_table[remaining_count][1]
                if 0 not in actions or np.random.random() < eps or blown_percentage < threshold :
                    action = 9
                    r_table[remaining_count][1] += 1
                else:
                    action = 0
            new_s, r, done, log = env.step(action)
            safe = new_s[7]
            if action == 9 and not safe:
                steps += 1
                r_table[remaining_count][0] += 1
            elif action == 0:
                steps += 1
            if log:
                print(log)
        all_steps.append(steps)
    return np.average(all_steps)

def predictive_potential(env, num_episodes=500):
    # given the number of dice i have left what is the average earning [count, total]
    r_table = np.zeros((6, 2))
    lr = 0.8

    all_steps = []
    eps = 0.5
    decay_factor = 0.999
    for g in range(num_episodes):
        s = env.reset()
        done = False
        steps = 0
        eps *= decay_factor
        pre_banked_points = np.array([])
        num_die_left_list = []
        turn_logs = []
        while not done:
            actions = np.array(env.legal_actions())
            num_die_left = env.num_remaining_dice()
            point_actions = np.setdiff1d(actions, [0,9])
            turn_logs.append(["prebanked: " + str(pre_banked_points), "die left: " + str(num_die_left), "available actions: " + str(actions)])
            if np.random.random() < eps:
               action = np.random.choice(actions)
            elif len(actions) == 1:
                turn_logs[-1].append("choice -z")
                action = actions[0]
            elif 9 in actions:
                turn_logs[-1].append("choice a")
                # here we have an option to roll or take more
                count, total = r_table[num_die_left-1]
                max_action = max(actions, key=you_blew_it.score_for_action)
                average_return = total / count if count != 0 else num_die_left * 100
                turn_logs[-1].append("average_return: {}, max_action: {}".format(average_return, max_action))
                # print "a", num_die_left, average_return, you_blew_it.score_for_action(max_action)
                if you_blew_it.score_for_action(max_action) > average_return:
                    action = max_action
                else:
                    # print "b"
                    action = 9
            # elif 9 in actions and len(point_actions) == 0:
            #     print "c"
            #     action = 9
            elif 0 in actions and len(point_actions) == 0:
                turn_logs[-1].append("choice b")
                # print "d"
                action = 0
            else:
                turn_logs[-1].append("choice c")
                # print "e"
                action = max(actions, key=you_blew_it.score_for_action)
            # print "step", action
            turn_logs[-1].append("chosen action: {}".format(action))
            new_s, r, done, log = env.step(action)
            blown = new_s[7]
            turn_logs[-1].append("old state: {}".format(s))
            turn_logs[-1].append("new state: {}".format(new_s))
            s = new_s

            if r != 0:
                pre_banked_points = np.append(pre_banked_points, [0])

                num_die_left = env.num_remaining_dice()
                count, total = r_table[num_die_left-1]
                new_average_return = total / count if count != 0 else num_die_left * 100

                pre_banked_points = np.add(pre_banked_points, [r + lr*(y*new_average_return - average_return)])
                num_die_left_list.append(num_die_left)
            if action == 9 and blown:
                # print json.dumps(turn_logs)
                turn_logs = []
                # blew it case
                steps += 1
                if num_die_left_list:
                    r_table[num_die_left-1][0] += 1
                    r_table[num_die_left-1][1] -= np.average(pre_banked_points)
                num_die_left_list = []
                pre_banked_points = np.array([])
            elif action == 0:
                # print json.dumps(turn_logs)
                turn_logs = []
                for i, count in enumerate(num_die_left_list):
                    r_table[count-1][0] += 1
                    r_table[count-1][1] += pre_banked_points[i]
                steps += 1
                num_die_left_list = []
                pre_banked_points = np.array([])
            if log:
                print json.dumps(turn_logs)
                print(log)
        all_steps.append(steps)
    for row in r_table:
        print row, row[1]/row[0]
    return np.average(all_steps)

def learn_from_scratch(env, num_episodes=5000000):
    # given the number of dice i have left what is the average earning [count, total]
    r_table = np.zeros((env.observation_space.n, env.action_space.n))
    lr = 0.8
    discount_rate = 0.99
    action_dict = {}
    all_steps = np.zeros(10, dtype=int)
    eps = 0.9
    decay_factor = 0.999
    games_failed = 0
    for g in range(num_episodes):
        if g % 10000 == 0:
            print g
        s = env.reset()
        actions = env.action_space
        done = False
        steps = 0
        eps *= decay_factor
        while not done:
            if np.random.random() < eps:
                # print "random"
                action = env.action_space.sample()
            else:
                # print "nr", r_table[s,:], np.argmax(r_table[s,:])
                action = np.argmax(r_table[s,:])
            action_dict[action] = action_dict.get(action, 0) + 1
            new_s, r, done, log = env.step(action)
            # print "0",steps, action
            # print "1",new_s
            # print "2",r_table[new_s,:]
            # print "3",np.argmax(r_table[new_s,:])
            updated = (r_table[s,action] * (1 - lr) +
                lr * (r + discount_rate * np.max(r_table[new_s,:])))
            # print "4",updated
            r_table[s, action] = updated
            s = new_s
            steps += 1
            if log:
                games_failed += 1
        all_steps[g % 10] = steps
    # for row in r_table:
    #     print row, row[1]/row[0]
    print games_failed / num_episodes
    print action_dict
    print r_table
    print all_steps
    return np.average(all_steps)

# model = Sequential()
# model.add(InputLayer(batch_input_shape=(1,8)))
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dense(10, activation='linear'))
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])

def keras_model(env, num_episodes=1):
    # given the number of dice i have left what is the average earning [count, total]
    r_table = np.zeros((6, 2))
    y = 0.95

    all_steps = []
    eps = 0.5
    decay_factor = 0.999
    for g in range(num_episodes):
        s = env.reset()
        done = False
        steps = 0
        eps *= decay_factor
        pre_banked_points = np.array([])
        num_die_left_list = []
        if g % 100 == 0:
            print("Episode {} of {}".format(g + 1, num_episodes))
        while not done:
            actions = np.array(env.legal_actions())
            num_die_left = env.num_remaining_dice()
            point_actions = np.setdiff1d(actions, [0,9])
            if np.random.random() < eps:
               action = np.random.choice(actions)
            else:
                action = np.argmax(model.predict(*s))
            # print "step", action
            new_s, r, done, log = env.step(action)
            blown = new_s[7]

            if r != 0:
                pre_banked_points = np.append(pre_banked_points, [0])
                target = r + y * model.predict(new_s)
                pre_banked_points = np.add(pre_banked_points, [target])
                target_vec = model.predict(s)[0]
                target_vec[action] = target
                model.fit(s, target_vec.reshape(-1, 10), epochs=1, verbose=0)
                num_die_left_list.append(num_die_left)
            if action == 9 and blown:
                # blew it case
                steps += 1
                if num_die_left_list:
                    target = r + y * model.predict(new_s)
                    target_vec = model.predict(s)[0]
                    target_vec[action] = np.average(pre_banked_points) * -1
                    model.fit(s, target_vec.reshape(-1, 10), epochs=1, verbose=0)
                num_die_left_list = []
                pre_banked_points = np.array([])
            elif action == 0:
                for i, count in enumerate(num_die_left_list):
                    target = r + y * model.predict(new_s)
                    pre_banked_points = np.add(pre_banked_points, [target])
                    target_vec = model.predict(s)[0]
                    target_vec[count] = target
                    model.fit(s, target_vec.reshape(-1, 10), epochs=1, verbose=0)
                steps += 1
                num_die_left_list = []
                pre_banked_points = np.array([])
            if log:
                print(log)
            s = new_s

        all_steps.append(steps)
    for row in r_table:
        print row, row[1]/row[0]

    return np.average(all_steps)

# print(random_legal_moves(env))
# print(greedy_treshold(env))
env.seed(1)
print(learn_from_scratch(env))
# model.save("/Users/evanrichards/workspace/youBlewIt/model.h5")







