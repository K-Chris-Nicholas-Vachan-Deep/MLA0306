# -*- coding: utf-8 -*-
"""rl exp 25.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CiO-H7ePZhnYxYh7a__LkCjlflEK-7zy
"""

import numpy as np
import tensorflow as tf
import gym

# Define the Vanilla Policy Gradient Agent
class VPGAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.02):
        self.policy_network = self.build_policy_network(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_policy_network(self, state_dim, action_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
        return model

    def select_action(self, state):
        action_probs = self.policy_network.predict(np.array([state]))
        action = np.random.choice(len(action_probs[0]), p=action_probs[0])
        return action

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(np.array(states))
            action_masks = tf.one_hot(actions, len(action_probs[0]))
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(selected_action_probs) * advantages)

        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

# Define the stock market environment
class StockMarketEnv:
    def __init__(self, price_data):
        self.price_data = price_data
        self.current_step = 0
        self.initial_balance = 10000 # Initial investment balance
        self.balance = self.initial_balance
        self.stock_units = 0
        self.max_steps = len(price_data) - 1

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_units = 0
        return [self.balance, self.stock_units]

    def step(self, action):
        if self.current_step >= self.max_steps:
            return [self.balance, self.stock_units], 0, True

        current_price = self.price_data[self.current_step]
        next_price = self.price_data[self.current_step + 1]

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.stock_units += 1
                self.balance -= current_price
        elif action == 0:  # Sell
            if self.stock_units > 0:
                self.stock_units -= 1
                self.balance += current_price

        self.current_step += 1

        # Calculate reward based on portfolio value
        portfolio_value = self.balance + (self.stock_units * next_price)
        reward = portfolio_value - self.initial_balance

        done = (self.current_step == self.max_steps)
        return [portfolio_value, self.stock_units], reward, done

# Training function for the VPG agent
def train_vpg_agent(agent, env, num_episodes=1000):
    state_dim = 2  # State: [portfolio_value, stock_units]
    action_dim = 2  # Actions: [Buy (1), Sell (0)]

    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Training episode-by-episode
        agent.train(states, actions, rewards)

        print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}")

if __name__ == "__main__":
    # Generate sample price data (replace with actual stock data)
    price_data = np.random.uniform(50, 150, size=100)

    env = StockMarketEnv(price_data)
    agent = VPGAgent(2, 2)
    train_vpg_agent(agent, env, num_episodes=100)

pip install stable-baselines3

import gym
from stable_baselines3 import PPO

# Define a custom gym environment
class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Discrete(16)  # 4x4 grid
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.state = 0  # Initial state
        self.goal_state = 15  # Goal state
        self.obstacle_states = [5, 7, 11]  # Obstacle states

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:  # Up
            new_state = self.state - 4 if self.state >= 4 else self.state
        elif action == 1:  # Down
            new_state = self.state + 4 if self.state < 12 else self.state
        elif action == 2:  # Left
            new_state = self.state - 1 if self.state % 4 != 0 else self.state
        elif action == 3:  # Right
            new_state = self.state + 1 if self.state % 4 != 3 else self.state

        if new_state in self.obstacle_states:
            reward = -1
            done = False
        elif new_state == self.goal_state:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.state = new_state
        return self.state, reward, done, {}

# Create the environment
env = CustomEnv()

# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Evaluate the trained agent
obs = env.reset()
for _ in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

env.close()