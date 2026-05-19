"""
agents.py - Q-Learning and DQN agents for Snake
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from environment import N_ACTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Tabular Q-Learning ───────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, alpha=0.001, gamma=0.99, epsilon=1.0,
                 eps_min=0.01, eps_decay=0.995, seed=None):
        self.alpha = alpha; self.gamma = gamma; self.epsilon = epsilon
        self.eps_min = eps_min; self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = {}
        self.name = "Q-Learning"

    def _get_q(self, obs):
        k = tuple(int(x) for x in obs)
        if k not in self.q_table:
            self.q_table[k] = np.zeros(N_ACTIONS, dtype=np.float64)
        return self.q_table[k]

    def select_action(self, obs):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self._get_q(obs)))

    def update(self, obs, action, reward, next_obs, done):
        q = self._get_q(obs)
        q_next = 0.0 if done else np.max(self._get_q(next_obs))
        td = reward + self.gamma * q_next - q[action]
        q[action] += self.alpha * td
        return float(td)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.q_table = d["q_table"]; self.epsilon = d["epsilon"]


# ─── DQN ─────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, n_in=11, n_out=N_ACTIONS, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done): self.buf.append((s, a, r, s2, done))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,s2,d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32).to(DEVICE),
                torch.tensor(a, dtype=torch.long).to(DEVICE),
                torch.tensor(r, dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(s2), dtype=torch.float32).to(DEVICE),
                torch.tensor(d, dtype=torch.float32).to(DEVICE))
    def __len__(self): return len(self.buf)


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0,
                 eps_min=0.01, eps_decay=0.995, batch_size=64,
                 target_update=50, buffer_cap=100_000, seed=None):
        if seed is not None:
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        self.gamma = gamma; self.epsilon = epsilon
        self.eps_min = eps_min; self.eps_decay = eps_decay
        self.batch_size = batch_size; self.target_update = target_update
        self.steps_done = 0; self.name = "DQN (Neural Network)"; self.device = DEVICE

        self.policy_net = QNetwork().to(DEVICE)
        self.target_net = QNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_cap)

    def select_action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return int(self.policy_net(t).argmax().item())

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            targets = r + self.gamma * q_next * (1.0 - d)
        loss = self.loss_fn(q, targets)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path):
        torch.save({"policy": self.policy_net.state_dict(),
                    "target": self.target_net.state_dict(),
                    "epsilon": self.epsilon, "steps": self.steps_done}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon = ckpt["epsilon"]; self.steps_done = ckpt["steps"]
        self.target_net.eval()
