import numpy as np
import random
from collections import deque

from keras.src.layers import Conv2D, Flatten
from keras.src.losses import mean_squared_error
from keras.src.optimizers import SGD
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import reduce_mean
import gymnasium as gym
import tensorflow as tf
from PIL import Image
from keras.src.layers import MaxPooling2D


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
        self.model.load_weights(f'dqn_{model_name}_{episode}.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'dqn_{model_name}_{episode}.h5')

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
        self.model.load_weights(f'ddqn_{model_name}_{episode}.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'ddqn_{model_name}_{episode}.h5')

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


class DuelingDQN:
    def __init__(self, state_space_shape, num_actions, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Dueling Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
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

    def _build_model(self, layers):
        """
        Builds a model with the given layers.
        :param layers: layers for the model
        """
        input_layer = Input(shape=self.state_space_shape)

        x = input_layer
        for layer in layers:
            x = layer(x)

        v = Dense(1)(x)
        a = Dense(self.num_actions)(x)

        q = (v + (a - reduce_mean(a, axis=1, keepdims=True)))

        model = Model(inputs=input_layer, outputs=q)
        model.compile(Adam(lr=self.learning_rate), loss=MeanSquaredError())
        return model

    def build_model(self, layers):
        """
        Builds the main and target network with the given layers.
        :param layers: layers for the models
        """
        self.model = self._build_model(layers)
        self.target_model = self._build_model(layers)
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
        self.model.load_weights(f'duelingdqn_{model_name}_{episode}.h5')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'duelingdqn_{model_name}_{episode}.h5')

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


def preprocess_state(state):
    img = Image.fromarray(state)
    img2 = img.convert('L')

    img3 = np.array(img2, dtype=float)
    img3 /= 255
    return img3


'''
Za ovaa zadaca ne mozev nikako da ja startuvam igrata bidejki celo vreme mi dava error ALE namespace is not found.
Site mozni resenija sto gi najdov na internet gi probav no ne uspeav da go sredam problemot.
Kodot go napisav bez da mozam da go testiram no bi trebalo da bide otprilika vaka
'''

if __name__ == '__main__':
    env = gym.make('ALE/MsPacman-v5', render_mode='human')
    tf.keras.config.disable_interactive_logging()

    num_actions = env.action_space.n
    state_space_shape = env.observation_space.shape
    initial_epsilon = 1
    min_epsilon = 0.01
    epsilon_decay = 0.995
    epsilon = initial_epsilon
    num_episodes = 1000
    num_steps_per_episode = 100

    layers = [Conv2D(32, activation='relu'),
              MaxPooling2D(),
              Conv2D(16, activation='relu'),
              MaxPooling2D(),
              Flatten()]

    agent = DuelingDQN(state_space_shape=state_space_shape, num_actions=num_actions)
    agent.build_model(layers)

    # Training
    episode = 0
    for episode in range(episode, num_episodes):
        print(f'Episode: {episode + 1}')
        state, _ = env.reset()
        for step in range(num_steps_per_episode):
            state = preprocess_state(state)
            action = agent.get_action(state, epsilon)
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            new_state, reward, terminated, _, _ = env.step(action)
            agent.update_memory(state, action, reward, new_state, terminated)

        agent.train()
        if episode % 10 == 0:
            agent.update_target_model()

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
                state = preprocess_state(state)
                action = agent.get_action(state, epsilon=0.5)
                new_state, reward, terminated, _, _ = env.step(action)
                agent.update_memory(state, action, reward, new_state, terminated)
                state = preprocess_state(new_state)
                steps += 1
                current_reward += reward
            total_steps += steps
            total_reward += current_reward

        avg_steps = total_steps / iteration
        avg_reward = total_reward / iteration
        print(f'Number of iterations: {iteration}, average steps: {avg_steps}, average reward: {avg_reward}')
        iters[iteration] = f'Average steps: {avg_steps}, average reward: {avg_reward}'

    print(iters)
