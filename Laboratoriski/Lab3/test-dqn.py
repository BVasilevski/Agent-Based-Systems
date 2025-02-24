import gymnasium as gym
from keras.src.losses import mean_squared_error
from keras.src.optimizers import SGD
from deep_q_learning import *
from keras.src.models import Sequential
from keras.src.layers import *
import tensorflow as tf


def build_model(state_space_shape, num_actions):
    model = Sequential()
    model.add(Dense(32, input_shape=state_space_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(SGD(learning_rate=0.01), mean_squared_error)

    return model


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    tf.keras.config.disable_interactive_logging()

    num_actions = env.action_space.n
    state_space_shape = env.observation_space.shape
    model = build_model(state_space_shape, num_actions)
    target_model = build_model(state_space_shape, num_actions)

    agent = DQN(state_space_shape, num_actions, model, target_model)
    agent.load('model', episode=1000)
    iters = {}

    for iteration in [50]:  # [25]:#, 50]:
        total_steps = 0
        total_reward = 0
        for iter in range(iteration):
            steps = 0
            current_reward = 0
            state, _ = env.reset()
            env.render()

            terminated = False
            while terminated is False:
                action = agent.get_action(state, epsilon=0.5)
                new_state, reward, terminated, _, _ = env.step(action)
                agent.update_memory(state, action, reward, new_state, terminated)
                state = new_state
                steps += 1
                print(f'{iter + 1} - {steps}')
                current_reward += reward
            total_steps += steps
            total_reward += current_reward

        avg_steps = total_steps / iteration
        avg_reward = total_reward / iteration
        print(f'Number of iterations: {iteration}, average steps: {avg_steps}, average reward: {avg_reward}')
        iters[iteration] = f'Average steps: {avg_steps}, average reward: {avg_reward}'

    print(iters)
