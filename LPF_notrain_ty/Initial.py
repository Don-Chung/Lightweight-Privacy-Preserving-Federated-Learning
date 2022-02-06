import configparser
import json
import os
import socket
import sys

import numpy as np

parent_dir = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(parent_dir + "/para.ini")

# Parameter initialization
user_number = int(config.get("paras", "user_number"))  # Number of users
total_round = int(config.get("paras", "total_round"))  # Number of running rounds
r = int(config.get("paras", "r"))  # The number of rows in the model
c = int(config.get("paras", "c"))  # The number of columns in the model
port_u_a = int(config.get("paras", "port_u_a"))  # Socket connection port, users and S_alpha


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
