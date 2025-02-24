from stable_baselines3 import DDPG
import gymnasium as gym
import numpy as np

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')

    n_actions = env.action_space.shape[-1]

    model = DDPG(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    iters = {}
    for iteration in [50, 100]:
        total_reward = 0
        for iter in range(iteration):
            print('Iteration {}'.format(iter + 1))
            obs, _ = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, _, _ = env.step(action)
                env.render()
                total_reward += reward
        avg_reward = total_reward / iteration
        print(f'Number of iterations: {iteration}, average reward: {avg_reward}')
        iters[f'Number of iterations: {iteration}'] = f'Average reward: {avg_reward}'

    print(iters)

'''
Za 50 iteracii dobivam prosecna nagrada -245
Za 100 iteracii dobivam prosecna nagrada -263
'''
