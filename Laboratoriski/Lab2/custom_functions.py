from q_learning import *


def train(discount_factor, learning_rate, num_episodes, num_steps_per_episode, epsilon, env, num_states, num_actions):
    q_table = random_q_table(-1, 0, (num_states, num_actions))
    if not epsilon and not env:
        for episode in range(num_episodes):
            state, _ = env.reset()
            for step in range(num_steps_per_episode):
                action = get_action(env, q_table, state, epsilon)
                new_state, reward, terminated, _, _ = env.step(action)
                new_q = calculate_new_q_value(q_table,
                                              state, new_state,
                                              action, reward,
                                              learning_rate, discount_factor)
                q_table[state, action] = new_q
                state = new_state
    else:
        for episode in range(num_episodes):
            state, _ = env.reset()
            for step in range(num_steps_per_episode):
                action = get_best_action(q_table, state)
                new_state, reward, terminated, _, _ = env.step(action)
                new_q = calculate_new_q_value(q_table,
                                              state, new_state,
                                              action, reward,
                                              learning_rate, discount_factor)
                q_table[state, action] = new_q
                state = new_state
    return q_table


def test(q_table, iterations, env, state, epsilon):
    total_steps = 0
    total_reward = 0

    if not epsilon:
        for i in range(0, iterations):
            state, _ = env.reset()
            steps = 0
            reward_current = 0
            terminated = False
            while not terminated:
                new_action = get_best_action(q_table, state)
                print(new_action)
                state, reward, terminated, _, _ = env.step(new_action)
                reward_current += reward
                steps += 1
                env.render()
            total_steps += steps
            total_reward += reward_current
    else:
        for i in range(0, iterations):
            env.reset()
            steps = 0
            reward_current = 0
            terminated = False
            while not terminated:
                new_action = get_action(env, q_table, state, epsilon)
                state, reward, terminated, _, _ = env.step(new_action)
                reward_current += reward
                steps += 1
                env.render()
            total_steps += steps
            total_reward += reward_current

    return total_steps / iterations, total_reward / iterations
