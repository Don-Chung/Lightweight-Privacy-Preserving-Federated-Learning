import configparser
import json
import os
import socket
import sys

import numpy as np
import torch.nn as nn  # 指定torch.nn别名nn
import torch.nn.functional as F  # 引用神经网络常用函数包，不具有可学习的参数

parent_dir = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(parent_dir + "/para.ini")

# Parameter initialization
user_number = int(config.get("paras", "user_number"))  # Number of users
total_round = int(config.get("paras", "total_round"))  # Number of running rounds
port_u_a = int(config.get("paras", "port_u_a"))  # Socket connection port, users and S_alpha


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 1000)  # 784表示输入神经元数量，1000表示输出神经元数量
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  # Applies a softmax followed by a logarithm, output batch * classes tensor


def init_cnn_alpha():
    fc1_weight = np.zeros((1000, 784))
    fc1_bias = np.zeros((1000,))
    fc2_weight = np.zeros((500, 1000))
    fc2_bias = np.zeros((500,))
    fc3_weight = np.zeros((200, 500))
    fc3_bias = np.zeros((200,))
    fc4_weight = np.zeros((10, 200))
    fc4_bias = np.zeros((10,))
    return fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias, fc4_weight, fc4_bias

def para():
    return user_number, total_round


def dimension():
    return r, c


def port_user_alpha():
    return port_u_a


def get_ip():
    # 获取本机电脑名
    Name = socket.getfqdn(socket.gethostname())
    # 获取本机ip
    Addr = socket.gethostbyname(Name)
    return Addr


# 解析JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024 * 1024)

    return round(fsize, 2)


SizeSet = {}


# 每个客户的数据集大小
class DataSize:
    def size_of_data(worker_num):
        SizeSet = []
        for i in range(worker_num):
            tmp = get_FileSize('./data/users/' + str(i + 1) + '_user.pkl')
            SizeSet.append(tmp)
        return SizeSet