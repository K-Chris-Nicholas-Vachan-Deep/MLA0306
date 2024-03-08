Python 3.11.6 (v3.11.6:8b6ee5ba3b, Oct  2 2023, 11:18:21) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
... 
... # Define the grid world
... n_rows, n_cols = 3, 4
... grid_world = np.zeros((n_rows, n_cols))
... 
... # Define rewards
... rewards = {
...     (0, 3): 10,   # Cheese state
...     (1, 3): -10,  # Penalty state
... }
... 
... # Define discount factor
... gamma = 0.9
... 
... # Define actions
... actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
... action_names = ['Right', 'Left', 'Down', 'Up']
... 
... # Function to calculate the Bellman update for a state
... def bellman_update(i, j, action):
...     if (i, j) in rewards:
...         return rewards[(i, j)]
... 
...     total_reward = 0
...     for a, (di, dj) in enumerate(actions):
...         next_i, next_j = i + di, j + dj
...         if 0 <= next_i < n_rows and 0 <= next_j < n_cols:
...             total_reward += 0.25 * (grid_world[next_i, next_j] * gamma)
... 
...     return total_reward
... 
... # Perform the Bellman update for state values
... num_iterations = 100
... for _ in range(num_iterations):
...     new_grid_world = np.zeros((n_rows, n_cols))
...     for i in range(n_rows):
...         for j in range(n_cols):
...             new_grid_world[i, j] = max([bellman_update(i, j, a) for a in actions])
... 
...     grid_world = new_grid_world
... 
... # Calculate the optimal policy
... optimal_policy = np.empty((n_rows, n_cols), dtype=object)
... for i in range(n_rows):
...     for j in range(n_cols):
...         if (i, j) not in rewards:
...             policy_values = [bellman_update(i, j, a) for a in actions]
...             if any(policy_values):
...                 optimal_policy[i, j] = action_names[np.argmax(policy_values)]
... 
... # Print the optimal policy
... print("Optimal Policy:")
... for row in optimal_policy:
...     print(" | ".join(action if action else ' ' for action in row))
