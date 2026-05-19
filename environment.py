"""
environment.py — Snake RL Custom Gymnasium Environment
=======================================================
Grid:    GRID_W x GRID_H discrete cells
Actions: 0=UP  1=RIGHT  2=DOWN  3=LEFT (absolute)

State (11 binary features):
  [danger_straight, danger_right, danger_left,
   dir_up, dir_right, dir_down, dir_left,
   food_left, food_right, food_up, food_down]

Rewards:
  +10   eat food
  -10   die (wall or self-collision)
  -0.01 per step  (efficiency incentive)
  +0.1 / -0.1  distance shaping (closer/farther to food)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

GRID_W = 15
GRID_H = 15
N_STATES = 11
N_ACTIONS = 3

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DELTA = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
TURN_LEFT  = {UP: LEFT,  LEFT: DOWN,  DOWN: RIGHT, RIGHT: UP}

REWARD_FOOD    =  10.0
REWARD_DIE     = -10.0
REWARD_STEP    = -0.01
REWARD_CLOSER  =  0.1
REWARD_FARTHER = -0.1
MAX_STEPS_NO_FOOD = GRID_W * GRID_H * 2


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_w=GRID_W, grid_h=GRID_H):
        super().__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.rng = np.random.default_rng()
        self._init_game()

    def _init_game(self):
        mid_r, mid_c = self.grid_h // 2, self.grid_w // 2
        self.snake = deque([(mid_r, mid_c), (mid_r, mid_c-1), (mid_r, mid_c-2)])
        self.direction = RIGHT
        self.score = 0
        self.steps_since_food = 0
        self._place_food()

    def _place_food(self):
        body = set(self.snake)
        empty = [(r, c) for r in range(self.grid_h) for c in range(self.grid_w)
                 if (r, c) not in body]
        self.food = empty[self.rng.integers(len(empty))] if empty else None

    def _is_danger(self, direction):
        hr, hc = self.snake[0]
        dr, dc = DELTA[direction]
        nr, nc = hr + dr, hc + dc
        if nr < 0 or nr >= self.grid_h or nc < 0 or nc >= self.grid_w:
            return True
        return (nr, nc) in list(self.snake)[:-1]

    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def get_state(self):
        d = self.direction
        head = self.snake[0]
        s = [
            float(self._is_danger(d)),
            float(self._is_danger(TURN_RIGHT[d])),
            float(self._is_danger(TURN_LEFT[d])),
            float(d == UP), float(d == RIGHT),
            float(d == DOWN), float(d == LEFT),
        ]
        if self.food:
            s += [float(self.food[1] < head[1]), float(self.food[1] > head[1]),
                  float(self.food[0] < head[0]), float(self.food[0] > head[0])]
        else:
            s += [0.0, 0.0, 0.0, 0.0]
        return np.array(s, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_game()
        return self.get_state(), {}

    def step(self, action):
        opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action != opposite[self.direction]:
            self.direction = action

        head = self.snake[0]
        dr, dc = DELTA[self.direction]
        new_head = (head[0]+dr, head[1]+dc)
        r, c = new_head

        if r < 0 or r >= self.grid_h or c < 0 or c >= self.grid_w \
                or new_head in list(self.snake)[:-1]:
            return self.get_state(), REWARD_DIE, True, False, {"score": self.score}

        old_dist = self._manhattan(head, self.food) if self.food else 0
        self.snake.appendleft(new_head)
        self.steps_since_food += 1

        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()
            new_dist = self._manhattan(new_head, self.food) if self.food else 0
            reward = REWARD_STEP + (REWARD_CLOSER if new_dist < old_dist else REWARD_FARTHER)

        if self.steps_since_food >= MAX_STEPS_NO_FOOD:
            return self.get_state(), REWARD_DIE, True, False, {"score": self.score}

        return self.get_state(), reward, False, False, {"score": self.score}

    def render(self):
        grid = [["." for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        for r, c in list(self.snake)[1:]:
            grid[r][c] = "o"
        hr, hc = self.snake[0]
        grid[hr][hc] = "H"
        if self.food:
            grid[self.food[0]][self.food[1]] = "F"
        lines = ["+" + "-"*self.grid_w + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-"*self.grid_w + "+")
        print("\n".join(lines))
