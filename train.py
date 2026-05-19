"""
train.py — Training loops for Q-Learning and DQN agents on Snake
"""
import numpy as np
from environment import SnakeEnv


def _is_dqn(agent):
    return hasattr(agent, 'train_step')


def train_qlearning(agent, n_episodes=2000, seed=None):
    env = SnakeEnv()
    if seed is not None: env.reset(seed=seed)
    ep_rewards, ep_scores, ep_lengths = [], [], []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_r, steps = 0, 0
        while True:
            a = agent.select_action(state)
            ns, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.update(state, a, r, ns, done)
            state = ns; total_r += r; steps += 1
            if done: break
        agent.decay_epsilon()
        ep_rewards.append(total_r); ep_scores.append(info["score"]); ep_lengths.append(steps)
    return {"episode_rewards": np.array(ep_rewards),
            "episode_scores": np.array(ep_scores),
            "episode_lengths": np.array(ep_lengths), "losses": np.array([])}


def train_dqn(agent, n_episodes=2000, seed=None):
    env = SnakeEnv()
    if seed is not None: env.reset(seed=seed)
    ep_rewards, ep_scores, ep_lengths, ep_losses = [], [], [], []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_r, ep_loss, steps = 0, [], 0
        while True:
            a = agent.select_action(state)
            ns, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.remember(state, a, r, ns, float(done))
            loss = agent.train_step()
            if loss > 0: ep_loss.append(loss)
            state = ns; total_r += r; steps += 1
            if done: break
        agent.decay_epsilon()
        ep_rewards.append(total_r); ep_scores.append(info["score"])
        ep_lengths.append(steps); ep_losses.append(np.mean(ep_loss) if ep_loss else 0.0)
    return {"episode_rewards": np.array(ep_rewards),
            "episode_scores": np.array(ep_scores),
            "episode_lengths": np.array(ep_lengths), "losses": np.array(ep_losses)}


def evaluate_agent(agent, n_eval=100, is_dqn=False):
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    env = SnakeEnv()
    rewards, scores = [], []
    for _ in range(n_eval):
        state, _ = env.reset()
        total_r = 0
        while True:
            a = agent.select_action(state)
            ns, r, term, trunc, info = env.step(a)
            total_r += r; state = ns
            if term or trunc: break
        rewards.append(total_r); scores.append(info["score"])
    agent.epsilon = saved_eps
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(scores))


def multi_run(agent_factory, n_runs=5, n_episodes=2000, train_fn=None):
    all_scores, all_rewards = [], []
    for run_idx in range(n_runs):
        print(f"    run {run_idx+1}/{n_runs}...")
        agent = agent_factory(seed=run_idx * 77)
        result = train_fn(agent, n_episodes=n_episodes, seed=run_idx * 13)
        all_scores.append(result["episode_scores"])
        all_rewards.append(result["episode_rewards"])
    return np.array(all_scores), np.array(all_rewards)
