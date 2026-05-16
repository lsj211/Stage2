import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from QRobot import QRobot
from Maze import Maze
from ReplayDataSet import ReplayDataSet
from torch_py.QNetwork import QNetwork

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from QRobot import QRobot  # 假设 QRobot 已定义并可用

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def add(self, s, a, r, s_next, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = (s, a, r, s_next, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.memory)

class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Robot(QRobot):
    valid_action = ["u", "d", "l", "r"]

    def __init__(self, maze):
        super(Robot, self).__init__(maze)
        self.maze = maze
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 状态维度仍是2：x, y
        self.state_dim = 2
        self.action_dim = len(self.valid_action)

        # 超参数
        self.lr = 1e-3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update_freq = 200

        # 网络 & 优化器
        self.eval_net   = DQNNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_fn   = nn.MSELoss()

        # 经验回放
        self.memory = ReplayBuffer(5000)
        self.learn_step = 0

    def state_to_vec(self, state):
        # 直接用 (x, y) 两个浮点值，不做归一化
        return np.array(state, dtype=np.float32)

    def choose_action(self, state_vec):
        if random.random() < self.epsilon:
            return random.choice(self.valid_action)
        with torch.no_grad():
            st = torch.tensor([state_vec], dtype=torch.float32, device=self.device)
            qvals = self.eval_net(st)[0].cpu().numpy()
        return self.valid_action[qvals.argmax()]

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        s, a, r, s_next, done = self.memory.sample(self.batch_size)
        s      = torch.tensor(s, dtype=torch.float32, device=self.device)
        a      = torch.tensor(a, dtype=torch.long,    device=self.device).unsqueeze(1)
        r      = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s_next = torch.tensor(s_next, dtype=torch.float32, device=self.device)
        done   = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_eval = self.eval_net(s).gather(1, a)
        with torch.no_grad():
            q_next   = self.target_net(s_next).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1 - done)

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def train_update(self):
        # 感知并向量化状态
        state = self.sense_state()
        state_vec = self.state_to_vec(state)

        # 选择并执行动作
        action = self.choose_action(state_vec)
        reward = -1.0  # 基础步惩
        self.maze.move_robot(action)
        next_state = self.sense_state()
        next_vec = self.state_to_vec(next_state)

        # 简单的距离塑形：靠近目标 +0.5，远离-0.5
        cur_dist  = abs(state[0]-self.maze.destination[0]) + abs(state[1]-self.maze.destination[1])
        next_dist = abs(next_state[0]-self.maze.destination[0]) + abs(next_state[1]-self.maze.destination[1])
        reward += 0.5 if next_dist < cur_dist else -0.5

        done = (next_state == self.maze.destination)
        if done:
            reward += 100.0

        # 记录并学习
        a_idx = self.valid_action.index(action)
        self.memory.add(state_vec, a_idx, reward, next_vec, done)
        self.learn()

        # ε 衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 回合结束重置
        if done:
            self.reset()

        return action, reward

    def test_update(self):
        state = self.sense_state()
        state_vec = self.state_to_vec(state)
        with torch.no_grad():
            st = torch.tensor([state_vec], dtype=torch.float32, device=self.device)
            qvals = self.eval_net(st)[0].cpu().numpy()
        action = self.valid_action[qvals.argmax()]
        reward = -1.0
        self.maze.move_robot(action)
        next_state = self.sense_state()
        if next_state == self.maze.destination:
            reward += 100.0
            self.reset()
        return action, reward

"""  Deep Qlearning 算法相关参数： """
from Runner import Runner
epoch = 10  # 训练轮数
maze_size = 5  # 迷宫size
training_per_epoch=int(maze_size * maze_size * 1.5)

""" 使用 DQN 算法训练 """

g = Maze(maze_size=maze_size)
r = Robot(g)
runner = Runner(r)
runner.run_training(epoch, training_per_epoch)

# 生成训练过程的gif图, 建议下载到本地查看；也可以注释该行代码，加快运行速度。
runner.generate_gif(filename="results/dqn_size10.gif")

print("训练完成！")

a=1+1