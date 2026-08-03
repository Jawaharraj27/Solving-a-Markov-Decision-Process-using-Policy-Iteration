# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8

def policy_evaluation(policy, env, V, gamma=0.99, theta=1e-8):
    """
    Evaluates a policy until value convergence within precision threshold theta.
    """
    while True:
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            v_new = 0
            
            for a in range(n_actions):
                action_prob = policy[s, a]
                # Transition dynamics: env.P[s][a] returns tuples of (prob, next_state, reward, terminated)
                for prob, next_state, reward, done in env.P[s][a]:
                    v_new += action_prob * prob * (reward + gamma * V[next_state])
            
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break
            
    return V

def policy_improvement(policy, env, V, gamma=0.99):
    """
    Greedily improves the policy given state-value function V.
    Returns (policy, policy_stable).
    """
    policy_stable = True
    
    for s in range(n_states):
        old_action = np.argmax(policy[s])
        action_values = np.zeros(n_actions)
        
        # Calculate expected return for each action in state s
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        
        # Greedily select best action (one-hot encoding)
        best_action = np.argmax(action_values)
        policy[s] = np.eye(n_actions)[best_action]
        
        if old_action != best_action:
            policy_stable = False
            
    return policy, policy_stable


def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Runs full Policy Iteration to find the optimal policy and state-value function.
    """
    # Initialize value function to zeros
    V = np.zeros(n_states)
    
    # Initialize uniform random policy across actions
    policy = np.ones([n_states, n_actions]) / n_actions
    
    while True:
        # Step 1: Evaluate current policy
        V = policy_evaluation(policy, env, V, gamma, theta)
        
        # Step 2: Improve policy based on V
        policy, policy_stable = policy_improvement(policy, env, V, gamma)
        
        # If policy stopped changing, optimal policy is found
        if policy_stable:
            break
            
    return policy, V

def print_value_function(V):
    print("\nOptimal State-Value Function:")
    print(np.round(V.reshape(4, 4), 4))


def print_policy(policy):
    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)
    policy_grid = np.array(
        [action_symbols[action] for action in best_actions]
    ).reshape(4, 4)

    print("\nOptimal Policy:")
    print(policy_grid)

optimal_policy, optimal_value_function = policy_iteration(
    env,
    gamma=gamma,
    theta=theta
)

print("Name: JAWAHAR RAJ N")
print("Register Number: 212223240057")
print_value_function(optimal_value_function)
print_policy(optimal_policy)

env.close()
```

## Output

<img width="1068" height="343" alt="image" src="https://github.com/user-attachments/assets/d14444c3-9e0a-44d1-9bc8-88bf478f4d93" />




---

## Result

The Policy Iteration algorithm was successfully implemented to obtain the optimal state-value function and optimal policy for the FrozenLake-v1 environment.


## Inference

The Policy Iteration algorithm successfully found the optimal policy and optimal state-value function for the FrozenLake environment. By changing the discount factor to γ = 0.99, the algorithm still converged successfully, but the state-value function changed slightly because future rewards were given a little less importance. The optimal policy remained the same for this environment.

