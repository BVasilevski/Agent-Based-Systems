import numpy as np
import gymnasium as gym
from mdp import value_iteration, policy_iteration

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi')

    state, _ = env.reset()
    env.render()
    terminated = False

    total_rewards = {}
    for discount in [0.5, 0.7, 0.9]:
        state, _ = env.reset()
        env.reset()
        terminated = False
        policy, _ = value_iteration(env,
                                    env.action_space.n,
                                    env.observation_space.n, discount_factor=discount)
        total_reward = 0
        while not terminated:
            new_action = np.argmax(policy[state])
            state, reward, terminated, _, _ = env.step(new_action)
            total_reward += reward
            env.render()
        total_rewards[f'Discount: {discount}'] = total_reward

    print("Total reward per discount:")
    print(total_rewards)
    max_reward_discount = str(max(total_rewards, key=total_rewards.get)).split(" ")[1]
    discount = float(max_reward_discount)

    best_policy, _ = value_iteration(env,
                                     env.action_space.n,
                                     env.observation_space.n, discount_factor=discount)
    iter_test = {}
    for iterations in [50, 100]:
        total_steps = 0
        total_reward = 0
        for i in range(0, iterations):
            env.reset()
            steps = 0
            reward_current = 0
            terminated = False
            while not terminated:
                new_action = np.argmax(best_policy[state])
                state, reward, terminated, _, _ = env.step(new_action)
                reward_current += reward
                steps += 1
                env.render()
            total_steps += steps
            total_reward += reward_current
        iter_test[
            f'{iterations} iterations'] = f'Average steps: {total_steps / iterations}, Average reward: {total_reward / iterations}'

    print("Average steps and average reward per iterations:")
    print(iter_test)
