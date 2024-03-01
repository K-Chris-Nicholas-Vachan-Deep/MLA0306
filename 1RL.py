import numpy as np

num_rows = 3
num_cols = 4
num_actions = 4  # Up, Down, Left, Right

rewards = np.zeros((num_rows, num_cols))
rewards[0, 3] = 1  # Goal with reward 1
rewards[1, 3] = -1  # Fire with reward -1
rewards[2, 3] = -1  # Fire with reward -1

start_row, start_col = 0, 0

Q = np.zeros((num_rows, num_cols, num_actions))

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon for epsilon-greedy strategy
num_episodes = 1000  # Number of episodes

def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)  # Explore
    else:
        return np.argmax(Q[state[0], state[1]])  # Exploit

def take_step(state, action):
    if action == 0:  # Up
        next_state = (max(state[0] - 1, 0), state[1])
    elif action == 1:  # Down
        next_state = (min(state[0] + 1, num_rows - 1), state[1])
    elif action == 2:  # Left
        next_state = (state[0], max(state[1] - 1, 0))
    else:  # Right
        next_state = (state[0], min(state[1] + 1, num_cols - 1))
    reward = rewards[next_state[0], next_state[1]]
    return next_state, reward

for _ in range(num_episodes):
    state = (start_row, start_col)
    total_reward = 0

    while True:
        action = choose_action(state)
        next_state, reward = take_step(state, action)
        total_reward += reward

        best_next_action = np.argmax(Q[next_state[0], next_state[1]])
        td_target = reward + gamma * Q[next_state[0], next_state[1], best_next_action]
        td_error = td_target - Q[state[0], state[1], action]
        Q[state[0], state[1], action] += alpha * td_error

        state = next_state

        if reward == 1 or reward == -1:  # Goal or Fire reached
            break

print("Q-table:")
print(Q)

state = (start_row, start_col)
path = [(state[0], state[1])]
total_reward = 0

while True:
    action = np.argmax(Q[state[0], state[1]])
    next_state, reward = take_step(state, action)
    total_reward += reward
    path.append((next_state[0], next_state[1]))

    state = next_state

    if reward == 1 or reward == -1:  # Goal or Fire reached
        break

print("Path to goal:")
for row, col in path:
    print(f"({row + 1}, {col + 1})")

print("Total Reward:", total_reward)
