import random
import time
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

ENV_NAME = 'CartPole-v1'
SEED = 24
ITERATIONS = 1000
BATCH_SIZE = 32
MODE = 'train'  # modes: train, test


class DQN:
    def __init__(self, states, actions, gamma=0.99, memory=deque(maxlen=100000), exploration_rate=1.0,
                 exploration_decay=0.99, exploration_min=1e-2, learning_rate=1e-3):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.memory = memory
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.states, activation="relu", kernel_initializer="RandomUniform"))
        model.add(Dense(self.actions, activation="linear", kernel_initializer="RandomUniform"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_action, done):
        self.memory.append((state, action, reward, next_action, done))

    def choose_action(self, state):
        state = np.reshape(state, [1, self.states])
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values) if np.random.random() > self.exploration_rate else random.randrange(self.actions)

    def replay(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            state = np.reshape(state, [1, env.observation_space.shape[0]])
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward
            targets = self.model.predict(state)
            targets[0][action] = target
            self.model.fit(state, targets, epochs=1, verbose=0)
            self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.exploration_min)

    def mode(self):
        if MODE == 'test':
            self.exploration_rate = self.exploration_min

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    np.random.seed(SEED)
    env.seed(SEED)
    actions = env.action_space.n
    states = env.observation_space.shape[0]
    agent = DQN(states, actions)
    scores, durations, iteration = [], [], []
    if MODE == 'test':
        agent.exploration_rate = agent.exploration_min
        agent.load_weights(('./model/dqn_{}_weights.h5f'.format(ENV_NAME)))
    for i in range(ITERATIONS):
        state = env.reset()
        score = 0
        for t in range(600):
            if MODE == 'test':
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            agent.memorize(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                print("Iteration: {}/{}, Score: {}, Memory: {}".format(i + 1, ITERATIONS, t, len(agent.memory)))
                scores.append(score)
                durations.append(t)
                iteration.append(i)
                break
            if len(agent.memory) > BATCH_SIZE and MODE == 'train':
                agent.replay()

    if MODE == 'train':
        agent.save_weights('./model/dqn_{}_weights.h5f'.format(ENV_NAME))

        # Plot Diagram
        plt.plot(iteration, scores)
        plt.xlabel("Iterations")
        plt.ylabel("Reward")
        plt.title("Rewards In Training")
        plt.savefig('./img/dqn_{}-{}.png'.format(ENV_NAME, time.strftime("%Y%m%d-%H%M%S")))
