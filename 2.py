import random
from QRobot import QRobot
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本

# class Robot(QRobot):
#     valid_action = ['u', 'r', 'd', 'l']
#
#     def __init__(self, maze):
#         super(Robot, self).__init__(maze)
#         self.maze = maze
#
#         # 1. 初始化 Q‑表
#         self.q_table = {}  # 键：state 元组 (r,c)，值：{ 'u':Q, 'r':Q, 'd':Q, 'l':Q }
#
#         # 2. Q‑learning 参数
#         self.alpha = 0.1  # 学习率
#         self.gamma = 0.9  # 折扣因子
#         self.epsilon = 0.4  # 初始探索概率 ε
#
#         # 3. 设置奖励：撞墙 −10，终点 +100，正常步进 −0.1
#         self.maze.set_reward({
#             "hit_wall": -10.0,
#             "destination": +100.0,
#             "default": -0.1
#         })
#
#     def train_update(self):
#         """
#         训练时调用：执行一步 Q-learning 更新，并返回 (action, reward)。
#         """
#         state = self.maze.sense_robot()  # 当前位置 (r,c)
#
#         # 如果 Q‑表里没有该状态，就初始化
#         if state not in self.q_table:
#             self.q_table[state] = {a: 0.0 for a in self.valid_action}
#
#         # ε‑greedy 选动作
#         if random.random() < self.epsilon:
#             action = random.choice(self.valid_action)
#         else:
#             # 选 Q 值最大的动作；如果并列，就随机挑一个
#             max_q = max(self.q_table[state].values())
#             max_actions = [a for a, q in self.q_table[state].items() if q == max_q]
#             action = random.choice(max_actions)
#
#         # 执行动作并观察 reward、下一个状态
#         reward = self.maze.move_robot(action)
#         next_state = self.maze.sense_robot()
#
#         # 如果下一个状态不在表里，先初始化
#         if next_state not in self.q_table:
#             self.q_table[next_state] = {a: 0.0 for a in self.valid_action}
#
#         # Q‑learning 核心更新
#         old_q = self.q_table[state][action]
#         # 下一个状态的最大 Q 值
#         next_max_q = max(self.q_table[next_state].values())
#         # 目标 target = r + γ * next_max_q
#         target = reward + self.gamma * next_max_q
#         # 学习率更新
#         self.q_table[state][action] = (1 - self.alpha) * old_q + self.alpha * target
#
#         # 衰减 ε（但不要太快，衰减得慢一些）
#         self.epsilon = max(0.05, self.epsilon * 0.995)
#
#         return action, reward
#
#     def test_update(self):
#         """
#         测试时调用：只选 Q‑值最大的动作（ε=0），不做更新，返回 (action, reward)。
#         """
#         state = self.maze.sense_robot()
#
#         # 如果表里没有，先初始化
#         if state not in self.q_table:
#             self.q_table[state] = {a: 0.0 for a in self.valid_action}
#
#         # 选 Q 值最大的动作
#         max_q = max(self.q_table[state].values())
#         max_actions = [a for a, q in self.q_table[state].items() if q == max_q]
#         action = random.choice(max_actions)
#
#         reward = self.maze.move_robot(action)
#         return action, reward

# if __name__ == "__main__":
#     maze = Maze(maze_size=5)
#     robot = Robot(maze)
#
#     num_episodes = 2000
#     max_steps_per_episode = 100
#
#     for ep in range(num_episodes):
#         maze.reset_robot()  # 每个 episode 开始时，把机器人放回 (0,0)
#         for step in range(max_steps_per_episode):
#             action, reward = robot.train_update()
#             state = maze.sense_robot()
#             if state == maze.destination:
#                 break
#
#     # 训练结束后，进行一次测试（纯贪心，不更新 Q‑表）
#     maze.reset_robot()
#     robot.epsilon = 0  # 确保测试阶段不再随机，只走贪心策略
#
#     for step in range(25):  # 最多 25 步，5×5 迷宫最长只需 8 步左右就能到达
#         action, reward = robot.test_update()
#         print(f"Step {step+1}: action={action}, reward={reward}, loc={maze.sense_robot()}")
#         if maze.sense_robot() == maze.destination:
#             print("成功到达终点！")
#             break
#     else:
#         print("没能到达终点……")




import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

from QRobot import QRobot
from Maze import Maze
from ReplayDataSet import ReplayDataSet
from torch_py.QNetwork import QNetwork


import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from QRobot import QRobot
from Maze import Maze
from ReplayDataSet import ReplayDataSet
from torch_py.QNetwork import QNetwork


class OptimizedDQNRobot(QRobot):
    valid_action = ['u', 'r', 'd', 'l']

    def __init__(self, maze: Maze):
        super(OptimizedDQNRobot, self).__init__(maze)
        self.maze = maze

        # ————（一）合理的奖励设计————
        self.maze.set_reward({
            "hit_wall": -5.0,
            "destination": 100.0,
            "default": -1.0,
        })

        # ————（二）DQN 网络 & Target 网络————
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_model = QNetwork(state_size=2, action_size=4, seed=0).to(self.device)
        self.target_model = QNetwork(state_size=2, action_size=4, seed=0).to(self.device)
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval()

        # ————（三）优化器与超参数————
        self.optimizer = optim.Adam(self.eval_model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_count = 0
        self.target_update_freq = 500

        # ————（四）ε-greedy 探索参数————
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 20000
        self.epsilon = self.epsilon_start

        # ————（五）回放缓冲区————
        max_size = max(self.maze.maze_size ** 2 * 3, 10000)
        self.memory = ReplayDataSet(max_size=int(max_size))

        # —— 状态归一化用到的迷宫尺寸 ——
        h, w, _ = self.maze.maze_data.shape
        self.H = float(h)
        self.W = float(w)

    def _state_to_tensor(self, state):
        r, c = state
        sr = r / (self.H - 1.0)
        sc = c / (self.W - 1.0)
        arr = np.array([sr, sc], dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _select_action_train(self, state):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.update_count / self.epsilon_decay)
        )
        legal_actions = self.maze.can_move_actions(state)
        legal_idx = [self.valid_action.index(a) for a in legal_actions]

        if random.random() < self.epsilon:
            return random.choice(legal_idx)
        else:
            state_t = self._state_to_tensor(state)
            with torch.no_grad():
                q_values = self.eval_model(state_t).cpu().numpy().flatten()
            mask = np.full((4,), -1e9, dtype=np.float32)
            mask[legal_idx] = 0.0
            filtered_q = q_values + mask
            return int(np.argmax(filtered_q))

    def _select_action_test(self, state):
        legal_actions = self.maze.can_move_actions(state)
        legal_idx = [self.valid_action.index(a) for a in legal_actions]
        state_t = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.eval_model(state_t).cpu().numpy().flatten()
        mask = np.full((4,), -1e9, dtype=np.float32)
        mask[legal_idx] = 0.0
        filtered_q = q_values + mask
        return int(np.argmax(filtered_q))

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        state_b, action_b, reward_b, next_state_b, done_b = self.memory.random_sample(self.batch_size)
        state_b = torch.from_numpy(state_b).to(self.device)
        action_b = torch.from_numpy(action_b).unsqueeze(1).to(self.device)
        reward_b = torch.from_numpy(reward_b).unsqueeze(1).to(self.device)
        next_state_b = torch.from_numpy(next_state_b).to(self.device)
        done_b = torch.from_numpy(done_b).unsqueeze(1).to(self.device)

        q_current = self.eval_model(state_b).gather(1, action_b)
        with torch.no_grad():
            next_q_values = self.target_model(next_state_b)
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
        target_q = reward_b + (1.0 - done_b.float()) * self.gamma * max_next_q

        loss = F.mse_loss(q_current, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())

    def train_update(self):
        state = self.sense_state()
        action_idx = self._select_action_train(state)
        action_str = self.valid_action[action_idx]

        reward = self.maze.move_robot(action_str)
        next_state = self.sense_state()
        done = (next_state == self.maze.destination)

        # —— 关键修改 ——
        # 原来是用 ndarray，会导致 unhashable。这里改为 tuple。
        s_arr = (state[0] / (self.H - 1.0), state[1] / (self.W - 1.0))
        ns_arr = (next_state[0] / (self.H - 1.0), next_state[1] / (self.W - 1.0))
        self.memory.add(s_arr, action_idx, float(reward), ns_arr, int(done))

        self._optimize_model()
        return action_str, float(reward)

    def test_update(self):
        state = self.sense_state()
        action_idx = self._select_action_test(state)
        action_str = self.valid_action[action_idx]
        reward = self.maze.move_robot(action_str)
        return action_str, float(reward)


if __name__ == "__main__":
    maze = Maze(maze_size=5)
    robot = OptimizedDQNRobot(maze)

    num_episodes = 800
    max_steps_per_episode = 100

    for ep in range(num_episodes):
        maze.reset_robot()
        for step in range(max_steps_per_episode):
            a, r = robot.train_update()
            if maze.sense_robot() == maze.destination:
                break
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes}, ε={robot.epsilon:.3f}")

    maze.reset_robot()
    robot.epsilon = 0
    print("---- 开始测试 ----")
    for step in range(25):
        a, r = robot.test_update()
        loc = maze.sense_robot()
        print(f"Step {step+1}: action={a}, reward={r}, loc={loc}")
        if loc == maze.destination:
            print("成功到达终点！")
            break
    else:
        print("未能到达终点...")
