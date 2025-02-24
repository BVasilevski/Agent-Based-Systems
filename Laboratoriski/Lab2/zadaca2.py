import gymnasium as gym
from custom_functions import *

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi')

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    iter_test_normal = {}
    iter_test_greedy = {}

    num_episodes = 10
    num_steps_per_episode = 10

    for discount_factor in [0.5, 0.9]:
        for learning_rate in [0.1, 0.01]:
            q_table = train(discount_factor, learning_rate, num_episodes, num_steps_per_episode, None, env,
                            num_states, num_actions)
            print("Finished training!")
            env = gym.make("Taxi-v3", render_mode='ansi')
            for iterations in [50, 100]:
                state, _ = env.reset()
                avg_steps, avg_reward = test(q_table, iterations, env, state, None)
                iter_test_normal[
                    f'Discount factor: {discount_factor}, Learning rate: {learning_rate}, Number of iterations: {iterations}'] = \
                    f'Average steps: {avg_steps}, Average reward: {avg_reward}'

    print("Testing without epsilon greedy")
    print(iter_test_normal)

    for discount_factor in [0.5, 0.9]:
        for learning_rate in [0.1, 0.01]:
            q_table = train(discount_factor, learning_rate, num_episodes, num_steps_per_episode, 0.5, env,
                            num_states, num_actions)

            for iterations in [50, 100]:
                state, _ = env.reset()
                avg_steps, avg_reward = test(q_table, iterations, env, state, 0.5)
                iter_test_greedy[
                    f'Discount factor: {discount_factor}, Learning rate: {learning_rate}, Number of iterations: {iterations}'] = \
                    f'Average steps: {avg_steps}, Average reward: {avg_reward}'

    print("Testing with epsilon greedy")
    print(iter_test_greedy)

'''
Vo ovaa zadacha nesto se slucuva pri treniranjeto bidejki posle pri testiranjeto dobivam ista sledna akcija i celo vreme taka 
do beskonecnost ili dodeka ne crashne celata aplikacija. 
'''