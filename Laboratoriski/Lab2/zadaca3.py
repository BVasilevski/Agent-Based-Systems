import gymnasium as gym
from q_learning import *


def calculate_new_q_value(q_table, old_state, new_state, action, reward, low_value, window_size, lr=0.1,
                          discount_factor=0.99):
    new_state = get_discrete_state(state, low_value, window_size)
    max_future_q = np.max(q_table[new_state])
    if isinstance(old_state, tuple):
        current_q = q_table[old_state + (action,)]
    else:
        current_q = q_table[old_state, action]
    return (1 - lr) * current_q + lr * (reward + discount_factor * max_future_q)


def test(q_table, iterations, env, state, epsilon, low, high):
    total_steps = 0
    total_reward = 0
    for i in range(0, iterations):
        state, _ = env.reset()
        steps = 0
        reward_current = 0
        terminated = False
        while not terminated:  # and steps <= 500:
            state = get_discrete_state(state, low, high)
            new_action = get_action(env, q_table, state, epsilon)
            state, reward, terminated, _, _ = env.step(new_action)
            reward_current += reward
            steps += 1
            env.render()
        print(f'Steps: {steps} Reward: {reward_current}')
        total_steps += steps
        total_reward += reward_current

    return total_steps / iterations, total_reward / iterations


def train(discount_factor, learning_rate, num_episodes, num_steps_per_episode, epsilon, env, state, low, high, do_decay,
          epsilon_min, epsilon_decay):
    if not do_decay:
        q_table = random_q_table(-1, 0, (observation_space_size + [num_actions]))
        for episode in range(num_episodes):
            state, _ = env.reset()
            for step in range(num_steps_per_episode):
                discrete_state = get_discrete_state(state, low, high)
                new_action = get_action(env, q_table, discrete_state, epsilon)
                new_state, reward, terminated, _, _ = env.step(new_action)
                new_q = calculate_new_q_value(q_table, discrete_state, new_state, new_action, reward,
                                              observation_space_low_value, observation_space_high_value,
                                              learning_rate, discount_factor)
                q_table[discrete_state, new_action] = new_q
                state = new_state
    else:
        q_table = random_q_table(-1, 0, (observation_space_size + [num_actions]))
        for episode in range(num_episodes):
            state, _ = env.reset()
            for step in range(num_steps_per_episode):
                discrete_state = get_discrete_state(state, low, high)
                new_action = get_action(env, q_table, discrete_state, epsilon)
                new_state, reward, terminated, _, _ = env.step(new_action)
                new_q = calculate_new_q_value(q_table, discrete_state, new_state, new_action, reward,
                                              observation_space_low_value, observation_space_high_value,
                                              learning_rate, discount_factor)
                q_table[discrete_state, new_action] = new_q
                state = new_state
            if epsilon > epsilon_min:
                epsilon -= epsilon_decay
    return q_table


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(int))


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='rgb_array')

    num_actions = env.action_space.n

    observation_space_size = [50, 50]
    observation_space_low_value = env.observation_space.low
    observation_space_high_value = env.observation_space.high
    observation_window_size = (observation_space_high_value - observation_space_low_value) / observation_space_size

    iter_test_normal = {}
    iter_test_decay = {}
    num_episodes = 100
    num_steps_per_episode = 10
    epsilon = 0.5
    epsilon_min = 0.1
    epsilon_decay = 0.05

    # Without decay
    for discount_factor in [0.5, 0.9]:
        for learning_rate in [0.1, 0.01]:
            state, _ = env.reset()
            q_table = train(discount_factor, learning_rate, num_episodes, num_steps_per_episode, epsilon, env, state,
                            observation_space_low_value, observation_space_high_value, False, epsilon_min,
                            epsilon_decay)
            for iterations in [50, 100]:
                # env = gym.make('MountainCar-v0', render_mode='human')
                state, _ = env.reset()
                avg_steps, avg_reward = test(q_table, iterations, env, state, 0.5, observation_space_low_value,
                                             observation_space_high_value)
                iter_test_normal[
                    f'Discount factor: {discount_factor}, Learning rate: {learning_rate}, Number of iterations: {iterations}'] = \
                    f'Average steps: {avg_steps}, Average reward: {avg_reward}'
    env.render()
    print(iter_test_normal)

    # With decay
    for discount_factor in [0.5, 0.9]:
        for learning_rate in [0.1, 0.01]:
            state, _ = env.reset()
            q_table = train(discount_factor, learning_rate, num_episodes, num_steps_per_episode, epsilon, env, state,
                            observation_space_low_value, observation_space_high_value, True, epsilon_min,
                            epsilon_decay)
            for iterations in [50, 100]:
                state, _ = env.reset()
                avg_steps, avg_reward = test(q_table, iterations, env, state, 0.5, observation_space_low_value,
                                             observation_space_high_value)
                iter_test_decay[
                    f'Discount factor: {discount_factor}, Learning rate: {learning_rate}, Number of iterations: {iterations}'] = \
                    f'Average steps: {avg_steps}, Average reward: {avg_reward}'
    env.render()
    print(iter_test_decay)

'''
Vo ovaa zadacha si proagja treniranjeto no potoa kolata nikogash ne stignuva do celta bez razlika
kolku vreme raboti aplikacijata
'''
