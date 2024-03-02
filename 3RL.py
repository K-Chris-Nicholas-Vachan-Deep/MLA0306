import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the environment
env = gym.make('CartPole-v1')

# Neural network for the agent
class PolicyNetwork(nn.Module):
    def _init_(self, input_size, output_size):
        super(PolicyNetwork, self)._init_()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Agent class
class Agent:
    def _init_(self, input_size, output_size):
        self.policy_network = PolicyNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.output_size = output_size  # Store output size

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probabilities = self.policy_network(state)
        probabilities = probabilities.detach().numpy()  # Convert to numpy array
        action = np.random.choice(np.arange(self.output_size), p=probabilities)
        return action

# Training loop
agent = Agent(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
num_episodes = 100

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.optimizer.zero_grad()
        state_tensor = torch.from_numpy(state).float()
        action_tensor = torch.tensor(action)
        reward_tensor = torch.tensor(reward)

        log_prob = torch.log(agent.policy_network(state_tensor)[action_tensor])
        loss = -log_prob * reward_tensor
        loss.backward()
        agent.optimizer.step()

        episode_reward += reward
        state = next_state

        if done:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward}")

env.close()
