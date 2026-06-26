import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class ReinforcementLearningAgent:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        loss = []
        for log_prob, R in zip(self.log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.cat(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

class TradingEnvironment:
    def __init__(self, prices, initial_cash=1000):
        self.prices = prices
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([self.cash, self.shares, self.prices[self.current_step]])

    def step(self, action):
        price = self.prices[self.current_step]
        if action == 0:  # Buy
            if self.cash >= price:
                self.shares += 1
                self.cash -= price
        elif action == 1:  # Sell
            if self.shares > 0:
                self.shares -= 1
                self.cash += price
        # Hold action does nothing

        self.current_step += 1
        if self.current_step >= len(self.prices):
            self.done = True

        next_state = self._get_state()
        reward = self.cash + self.shares * price - self.initial_cash
        return next_state, reward, self.done

if __name__ == '__main__':
    # Dummy price data
    prices = np.sin(np.linspace(0, 10, 100)) * 10 + 100

    # Environment and agent setup
    env = TradingEnvironment(prices)
    state_dim = 3
    action_dim = 3  # Buy, Sell, Hold
    agent = ReinforcementLearningAgent(state_dim, action_dim)

    # Training loop
    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_reward(reward)
            total_reward += reward
            state = next_state
            if done:
                break
        agent.update_policy()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")