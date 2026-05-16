# 导入相关包
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本

import matplotlib.pyplot as plt

""" 创建迷宫并展示 """
# maze = Maze(maze_size=10) # 随机生成迷宫
# # print(maze)
#
#
# import random
#
# rewards = [] # 记录每走一步的奖励值
# actions = [] # 记录每走一步的移动方向
#
# # 循环、随机移动机器人10次，记录下奖励
# for i in range(10):
#     valid_actions = maze.can_move_actions(maze.sense_robot())
#     action = random.choice(valid_actions)
#     rewards.append(maze.move_robot(action))
#     actions.append(action)
#
# print("the history of rewards:", rewards)
# print("the actions", actions)
#
# # 输出机器人最后的位置
# print("the end position of robot:", maze.sense_robot())

# 打印迷宫，观察机器人位置
# print(maze)



import numpy as np

# 机器人移动方向
move_map = {
    'u': (-1, 0), # up
    'r': (0, +1), # right
    'd': (+1, 0), # down
    'l': (0, -1), # left
}


# 迷宫路径搜索树
class SearchTree(object):


    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """

        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0


def expand(maze, is_visit_m, node):
    """
    拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
    :param maze: 迷宫对象
    :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
    :param node: 待拓展的叶子节点
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + maze.move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)


def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path



def my_search(maze):
    """
    对迷宫进行深度优先搜索
    :param maze: 待搜索的 maze 对象
    """
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    stack = [root]  # 使用栈而非队列，实现深度优先
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=int)  # 标记访问状态
    path = []

    while stack:
        current_node = stack.pop()
        if is_visit_m[current_node.loc]:
            continue  # 若访问过则跳过
        is_visit_m[current_node.loc] = 1

        if current_node.loc == maze.destination:
            path = back_propagation(current_node)
            break

        expand(maze, is_visit_m, current_node)

        # 由于栈是后进先出，为了保持和动作顺序一致，倒序加入子节点
        for child in reversed(current_node.children):
            stack.append(child)

    return path


maze = Maze(maze_size=10) # 从文件生成迷宫

path_2 = my_search(maze)
print("搜索出的路径：", path_2)

for action in path_2:
    maze.move_robot(action)


if maze.sense_robot() == maze.destination:
    print("恭喜你，到达了目标点")