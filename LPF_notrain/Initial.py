import configparser
import json
import os
import random
import socket
import sys

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
r = int(config.get("paras", "r"))  # The number of rows in the model
c = int(config.get("paras", "c"))  # The number of columns in the model
port_u_a = int(config.get("paras", "port_u_a"))  # Socket connection port, users and S_alpha
port_u_b = int(config.get("paras", "port_u_b"))  # Socket connection port, users and S_beta
port_a_b = int(config.get("paras", "port_a_b"))  # Socket connection port, S_alpha and S_beta


def para():
    return user_number, total_round


def dimension():
    return r, c


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


# def gen_secret(seed, worker_number):
#     np.random.seed(seed)
#     return np.random.randint(1, 100, size=(worker_number,))
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


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
