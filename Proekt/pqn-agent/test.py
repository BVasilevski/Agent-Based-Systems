import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 120)),
            nn.LayerNorm(120),
            nn.ReLU(),
            layer_init(nn.Linear(120, 84)),
            nn.LayerNorm(84),
            nn.ReLU(),
            layer_init(nn.Linear(84, env.action_space.n)),
        )

    def forward(self, x):
        return self.network(x)


def make_env(env_id, seed, idx, capture_video, run_name, render_mode):
    env = gym.make(env_id, render_mode=render_mode)

    if capture_video and idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)

    return env


env_id = "CartPole-v1"
capture_video = False
render_mode = 'rgb_array'
run_name = "test_run"
seed = 1
results = {}

for model in os.listdir('models'):
    total_reward = 0
    num_episodes = 100
    epsilon = 0.1
    env = make_env(env_id, seed, 0, capture_video, run_name, render_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(f"models/{model}", map_location=device))
    q_network.eval()

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0

        while not done:
            obs_tensor = torch.Tensor(obs).to(device)
            with torch.no_grad():
                q_values = q_network(obs_tensor)
                action = torch.argmax(q_values).cpu().numpy()

            if np.random.rand() < epsilon:
                action = env.action_space.sample()

            next_obs, reward, done, info, _ = env.step(action)
            episode_reward += reward
            obs = next_obs
            if episode_reward >= 500:
                break
            env.render()

        print(f"Model: {model} for episode {episode + 1} achieved total reward = {episode_reward}")
        total_reward += episode_reward

    print(f'Model: {model} achieved average reward for {num_episodes} episodes: {total_reward / num_episodes}')
    results[model] = f'Num episodes: {num_episodes}, avg reward: {total_reward / num_episodes}'

    env.close()

for model in results:
    print(f'Model: {model} achieved average reward {results[model]}')
