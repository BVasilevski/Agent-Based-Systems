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

    agent = DDQN(state_space_shape, num_actions, model, target_model)
    initial_epsilon = 1
    min_epsilon = 0.01
    epsilon_decay = 0.995
    epsilon = initial_epsilon
    num_episodes = 1000
    num_steps_per_episode = 100

    episode = 0
    for episode in range(episode, num_episodes):
        print(f'Episode: {episode + 1}')
        state, _ = env.reset()
        for step in range(num_steps_per_episode):
            action = agent.get_action(state, epsilon)
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            new_state, reward, terminated, _, _ = env.step(action)
            agent.update_memory(state, action, reward, new_state, terminated)

        agent.train()
        if episode % 10 == 0:
            agent.update_target_model()
        if episode % 100 == 0:
            agent.save('model', episode=episode)

    agent.save('model', episode=episode + 1)
    print(f'Final epsilon: {epsilon}')
    print("FINISHED TRAINING")
