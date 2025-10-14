
---

## 🧠 Reinforcement Learning (RL)

**Reinforcement Learning** is a type of Machine Learning where an **agent** learns to make **sequences of decisions**
by interacting with an **environment**, receiving **rewards or penalties**, and improving its policy over time.

Unlike supervised learning (where we learn from labeled data),
RL learns from **trial and error** — by exploring actions and observing results.

---

### 🧭 The Core Idea

An agent observes a **state** ( s ), takes an **action** ( a ), and receives a **reward** ( r ).
The goal is to learn a **policy** that maximizes **cumulative reward**.

[
\pi^*(s) = \arg\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t ,\middle|, \pi \right]
]

Where:

* ( \pi(s) ): policy — mapping from state to action
* ( \gamma ): discount factor (0 ≤ γ ≤ 1), determines how much future rewards are valued
* ( r_t ): reward at time step ( t )

---

### 🧩 Components of an RL System

| Component              | Description                                    |
| ---------------------- | ---------------------------------------------- |
| **Agent**              | The learner or decision maker                  |
| **Environment**        | The world the agent interacts with             |
| **State (s)**          | Current situation                              |
| **Action (a)**         | Move the agent can make                        |
| **Reward (r)**         | Feedback signal from environment               |
| **Policy (π)**         | Strategy for choosing actions                  |
| **Value Function (V)** | Expected future reward from a state            |
| **Q-Function (Q)**     | Expected future reward for a state-action pair |

---

### 🧮 Bellman Equation

The **Bellman Equation** expresses the relationship between the value of a state and the values of its possible next states:

[
V(s) = \mathbb{E}[r + \gamma V(s')]
]

For **Q-values** (state-action pairs):

[
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
]

This forms the foundation for **Q-Learning**.

---

### 🔁 Q-Learning Algorithm

**Goal:** Learn the optimal action-value function ( Q^*(s, a) ).

**Update rule:**

[
Q(s, a) \leftarrow Q(s, a) + \alpha , [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
]

Where:

* ( \alpha ): learning rate
* ( \gamma ): discount factor

---

### 💻 Example: Simple Gridworld

```python
import numpy as np
import random

# Gridworld setup
actions = ['up', 'down', 'left', 'right']
Q = {}  # state-action value table
states = [(i, j) for i in range(4) for j in range(4)]
for s in states:
    Q[s] = {a: 0 for a in actions}

# Hyperparameters
alpha, gamma, epsilon = 0.1, 0.9, 0.1

def get_next_state(state, action):
    i, j = state
    if action == 'up': i = max(i-1, 0)
    if action == 'down': i = min(i+1, 3)
    if action == 'left': j = max(j-1, 0)
    if action == 'right': j = min(j+1, 3)
    return (i, j)

def reward(state):
    return 1 if state == (3, 3) else 0  # goal at (3,3)

# Q-learning loop
for episode in range(1000):
    s = random.choice(states)
    while s != (3, 3):
        a = random.choice(actions) if random.random() < epsilon else max(Q[s], key=Q[s].get)
        s_next = get_next_state(s, a)
        r = reward(s_next)
        Q[s][a] += alpha * (r + gamma * max(Q[s_next].values()) - Q[s][a])
        s = s_next
```

After training, ( Q(s, a) ) contains learned values for optimal behavior.

---

### 🤖 Modern Reinforcement Learning

| Approach                  | Description                                        |
| ------------------------- | -------------------------------------------------- |
| **Q-Learning**            | Tabular value updates                              |
| **Deep Q-Networks (DQN)** | Neural nets approximate Q-values                   |
| **Policy Gradient**       | Directly optimize policy (e.g. REINFORCE)          |
| **Actor-Critic**          | Combines value + policy learning                   |
| **PPO, A3C, DDPG**        | Advanced stable algorithms used in robotics, games |

---

### 🎮 Real-world Applications

| Domain         | Example                          |
| -------------- | -------------------------------- |
| **Games**      | AlphaGo, Chess AI, Atari         |
| **Robotics**   | Walking, balancing, manipulation |
| **Finance**    | Automated trading                |
| **Healthcare** | Personalized treatment           |
| **Operations** | Resource allocation, logistics   |

---

### 🚀 Summary

| Aspect           | Description                                    |
| ---------------- | ---------------------------------------------- |
| **Type**         | Sequential decision making                     |
| **Goal**         | Maximize long-term reward                      |
| **Key Concept**  | Learn from environment interaction             |
| **Core Formula** | Bellman equation + Q-learning update           |
| **Libraries**    | `gymnasium`, `stable-baselines3`, `ray[rllib]` |

---

