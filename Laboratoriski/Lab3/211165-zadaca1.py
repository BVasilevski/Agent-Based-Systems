import gymnasium as gym
from keras.src.losses import mean_squared_error
from keras.src.optimizers import SGD
from keras.src.models import Sequential
from keras.src.layers import *
import tensorflow as tf

import numpy as np
import random
from collections import deque


class DQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
        :param model: Keras model
        :param target_model: Keras model
        :param learning_rate: learning rate
        :param discount_factor: discount factor
        :param batch_size: batch size
        :param memory_size: maximum size of the experience replay memory
        """
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.target_model = target_model
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to experience replay memory.
        :param state: current state
        :param action: performed action
        :param reward: reward received for performing action
        :param next_state: next state
        :param done: if episode has terminated after performing the action in the current state
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Synchronize the target model with the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random() + epsilon / self.num_actions
        if probability < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            return np.argmax(self.model.predict(state)[0])

    def load(self, model_name, episode):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.load_weights(f'dqn_{model_name}_{episode}.weights.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'dqn_{model_name}_{episode}.weights.h5')

    def train(self):
        """
        Performs one step of model training.
        """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        if isinstance(self.state_space_shape, tuple):
            states = np.zeros((batch_size,) + self.state_space_shape)
        else:
            states = np.zeros((batch_size, self.state_space_shape))
        actions = np.zeros((batch_size, self.num_actions))

        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            if done:
                max_future_q = reward
            else:
                if isinstance(self.state_space_shape, tuple):
                    next_state = next_state.reshape((1,) + self.state_space_shape)
                else:
                    next_state = next_state.reshape(1, self.state_space_shape)
                max_future_q = (reward + self.discount_factor *
                                np.amax(self.target_model.predict(next_state)[0]))
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            target_q = self.model.predict(state)[0]
            target_q[action] = max_future_q
            states[i] = state
            actions[i] = target_q

        self.model.train_on_batch(states, actions)


class DDQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Double Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
        :param model: Keras model
        :param target_model: Keras model
        :param learning_rate: learning rate
        :param discount_factor: discount factor
        :param batch_size: batch size
        :param memory_size: maximum size of the experience replay memory
        """
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.target_model = target_model
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to experience replay memory.
        :param state: current state
        :param action: performed action
        :param reward: reward received for performing action
        :param next_state: next state
        :param done: if episode has terminated after performing the action in the current state
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Synchronize the target model with the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random() + epsilon / self.num_actions
        if probability < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            return np.argmax(self.model.predict(state)[0])

    def load(self, model_name, episode):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.load_weights(f'ddqn_{model_name}_{episode}.weights.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'ddqn_{model_name}_{episode}.weights.h5')

    def train(self):
        """
        Performs one step of model training.
        """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        if isinstance(self.state_space_shape, tuple):
            states = np.zeros((batch_size,) + self.state_space_shape)
        else:
            states = np.zeros((batch_size, self.state_space_shape))
        actions = np.zeros((batch_size, self.num_actions))

        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            if done:
                max_future_q = reward
            else:
                if isinstance(self.state_space_shape, tuple):
                    next_state = next_state.reshape((1,) + self.state_space_shape)
                else:
                    next_state = next_state.reshape(1, self.state_space_shape)
                max_action = np.argmax(self.model.predict(next_state)[0])
                max_future_q = (reward + self.discount_factor *
                                self.target_model.predict(next_state)[0][max_action])
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            target_q = self.model.predict(state)[0]
            target_q[action] = max_future_q
            states[i] = state
            actions[i] = target_q

        self.model.train_on_batch(states, actions)


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
    initial_epsilon = 1
    min_epsilon = 0.01
    epsilon_decay = 0.995
    epsilon = initial_epsilon
    num_episodes = 1000
    num_steps_per_episode = 100

    # Training
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

    agent.save('model', episode=episode + 1)

    # Testing
    iters = {}
    for iteration in [50, 100]:
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
                current_reward += reward
            total_steps += steps
            total_reward += current_reward

        avg_steps = total_steps / iteration
        avg_reward = total_reward / iteration
        print(f'Number of iterations: {iteration}, average steps: {avg_steps}, average reward: {avg_reward}')
        iters[iteration] = f'Average steps: {avg_steps}, average reward: {avg_reward}'

    print(iters)

'''
Vo 50 iteracii dobiv prosecen broj na cekori 5294 i prosecna nagrada -5294
Vo 100 iteracii dobiv prosecen broj na cekori 5097 i prosecna nagrada -5097
'''
