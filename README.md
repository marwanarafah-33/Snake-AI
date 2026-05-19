# Snake AI 

## Structure
```
snake_rl/
├── environment.py     — Custom Snake Gymnasium environment (15×15)
├── agents.py          — Tabular Q-Learning & DQN (neural network) agents
├── train.py           — Training loops, evaluation, multi-run stability
├── plots.py           — All required diagnostic visualisations
├── main.py            — Full pipeline runner
├── checkpoints/       — Saved DQN model weights
└── plots/             — All generated diagnostic plots
```

## Quick Start
```bash
pip install numpy matplotlib gymnasium torch
cd snake_rl
python main.py
```

## Environment
| Property | Value |
|---|---|
| Grid | 15×15 discrete cells |
| State | 11-bit binary feature vector |
| Actions | UP / RIGHT / DOWN / LEFT (4 absolute) |
| Food reward | +10 |
| Death reward | −10 (wall or self-collision) |
| Step penalty | −0.01 |
| Distance shaping | ±0.1 (closer / farther from food) |
| Starvation limit | 2 × GRID_W × GRID_H steps without food |

## State Representation (11 features)
```
[danger_straight, danger_right, danger_left,
 dir_up, dir_right, dir_down, dir_left,
 food_left, food_right, food_up, food_down]
```

## Algorithms
| Agent | Type | Key Details |
|---|---|---|
| **Q-Learning** | Tabular, off-policy TD | Dict-based Q-table, 11-bit state key, α=0.001 |
| **DQN** | Neural network, off-policy | Linear(11)→ReLU→Linear(256)→ReLU→Linear(4), replay buffer 100k, target net |

## Plots Generated
| File | Description |
|---|---|
| `00_snake_env.png` | Environment layout visualisation |
| `01_learning_curves.png` | Score & reward per episode + moving average |
| `02_value_heatmap.png` | Max Q-value distribution per agent |
| `03_policy_grid.png` | Greedy policy for representative states |
| `04_stability.png` | Mean ± std across 5 independent runs |
| `05_convergence.png` | Episodes to reach score threshold |
| `06_dqn_loss.png` | DQN Huber loss over training |
| `07_epsilon_decay.png` | Exploration decay schedule |
| `08_final_metrics.png` | Final evaluation bar charts |
| `09_score_distribution.png` | Score histogram (last 500 episodes) |
