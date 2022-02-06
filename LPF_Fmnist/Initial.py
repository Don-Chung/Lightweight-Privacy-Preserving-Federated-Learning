import configparser
import json
import os
import random
import socket
import sys

import torch.nn as nn  # 指定torch.nn别名nn
import torch.nn.functional as F  # 引用神经网络常用函数包，不具有可学习的参数
import numpy as np

from KeyAgreement import calculation

parent_dir = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(parent_dir + "/para.ini")
p = 109  # modulus
g = 6  # generator

# Parameter initialization
user_number = int(config.get("paras", "user_number"))  # Number of users
total_round = int(config.get("paras", "total_round"))  # Number of running rounds
dynamic = int(config.get("paras", "dynamic"))
dynamic_range = int(config.get("paras", "dynamic_range"))

port_u_a = int(config.get("paras", "port_u_a"))  # Socket connection port, users and S_alpha
port_u_b = int(config.get("paras", "port_u_b"))  # Socket connection port, users and S_beta
port_a_b = int(config.get("paras", "port_a_b"))  # Socket connection port, S_alpha and S_beta


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
    fc1_weight = np.zeros((1000, 392))
    fc1_bias = np.zeros((500,))
    fc2_weight = np.zeros((500, 500))
    fc2_bias = np.zeros((250,))
    fc3_weight = np.zeros((200, 250))
    fc3_bias = np.zeros((100,))
    fc4_weight = np.zeros((10, 100))
    fc4_bias = np.zeros((5,))
    return fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias, fc4_weight, fc4_bias


def init_cnn_beta():
    fc1_weight = np.zeros((1000, 392))
    fc1_bias = np.zeros((500,))
    fc2_weight = np.zeros((500, 500))
    fc2_bias = np.zeros((250,))
    fc3_weight = np.zeros((200, 250))
    fc3_bias = np.zeros((100,))
    fc4_weight = np.zeros((10, 100))
    fc4_bias = np.zeros((5,))
    return fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias, fc4_weight, fc4_bias


def para():
    return user_number, total_round


def para_dynamic():
    return dynamic, dynamic_range


# def dimension():
#     return r, c


def port_user_alpha():
    return port_u_a


def port_user_beta():
    return port_u_b


def port_alpha_beta():
    return port_a_b


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


def random_int_list(start, stop, length, seed):
    random.seed(seed)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


# create seed for users in round 0
def gen_secret(worker_number, private_key, public_key):
    seed = calculation(private_key, p, public_key)
    return random_int_list(1, 100, worker_number, seed)
    # return np.random.randint(1, 100, size=(worker_number,))


def gen_secret_join(user_index, private_key, public_key):
    seed = calculation(private_key, p, public_key) + user_index
    random.seed(seed)
    return random.randint(1, 100)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def dynamic_iod(dynamic_seed):
    random.seed(dynamic_seed)
    return random.randint(0, 1)


def dynamic_number(dynamic_seed, user_number_limit, user_number):
    random.seed(dynamic_seed)
    if user_number > user_number_limit:
        return random.randint(1, user_number - user_number_limit)
    if user_number < user_number_limit:
        return random.randint(1, user_number_limit - user_number)
    if user_number_limit == user_number:
        return 0


# def quit_id(dynamic_seed, num):
#     random.seed(dynamic_seed)
#     return random.randint(0, num - 1)
