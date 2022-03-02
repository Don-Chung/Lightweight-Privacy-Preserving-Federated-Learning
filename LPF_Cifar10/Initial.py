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


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Net(nn.Module):
    def __init__(self, vgg_name):
        super(Net, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11():
    return Net('VGG11')


def VGG13():
    return Net('VGG13')


def VGG16():
    return Net('VGG16')


def VGG19():
    return Net('VGG19')


def init_cnn_alpha():
    # 0
    # features_0_weight = np.zeros((192, 9))
    features_0_weight = np.zeros((192, 4))
    features_0_bias = np.zeros((32,))
    features_1_weight = np.zeros((32,))
    features_1_bias = np.zeros((32,))
    # 4
    # features_3_weight = np.zeros((, 9))
    features_3_weight = np.zeros((4096, 4))
    features_3_bias = np.zeros((32,))
    features_4_weight = np.zeros((32,))
    features_4_bias = np.zeros((32,))
    # 8
    # features_7_weight = np.zeros((, 9))
    features_7_weight = np.zeros((8192, 4))
    features_7_bias = np.zeros((64,))
    features_8_weight = np.zeros((64,))
    features_8_bias = np.zeros((64,))
    # 12
    # features_10_weight = np.zeros((16384, 9))
    features_10_weight = np.zeros((16384, 4))
    features_10_bias = np.zeros((64,))
    features_11_weight = np.zeros((64,))
    features_11_bias = np.zeros((64,))
    # 16
    # features_14_weight = np.zeros((32768, 9))
    features_14_weight = np.zeros((32768, 4))
    features_14_bias = np.zeros((128,))
    features_15_weight = np.zeros((128,))
    features_15_bias = np.zeros((128,))
    # 20
    # features_17_weight = np.zeros((65536, 9))
    features_17_weight = np.zeros((65536, 4))
    features_17_bias = np.zeros((128,))
    features_18_weight = np.zeros((128,))
    features_18_bias = np.zeros((128,))
    # 24
    # features_20_weight = np.zeros((65536, 9))
    features_20_weight = np.zeros((65536, 4))
    features_20_bias = np.zeros((128,))
    features_21_weight = np.zeros((128,))
    features_21_bias = np.zeros((128,))
    # 28
    # features_24_weight = np.zeros((131072, 9))
    features_24_weight = np.zeros((131072, 4))
    features_24_bias = np.zeros((256,))
    features_25_weight = np.zeros((256,))
    features_25_bias = np.zeros((256,))
    # 32
    # features_27_weight = np.zeros((262144, 9))
    features_27_weight = np.zeros((262144, 4))
    features_27_bias = np.zeros((256,))
    features_28_weight = np.zeros((256,))
    features_28_bias = np.zeros((256,))
    # 36
    # features_30_weight = np.zeros((262144, 9))
    features_30_weight = np.zeros((262144, 4))
    features_30_bias = np.zeros((256,))
    features_31_weight = np.zeros((256,))
    features_31_bias = np.zeros((256,))
    # 40
    # features_34_weight = np.zeros((262144, 9))
    features_34_weight = np.zeros((262144, 4))
    features_34_bias = np.zeros((256,))
    features_35_weight = np.zeros((256,))
    features_35_bias = np.zeros((256,))
    # 44
    # features_37_weight = np.zeros((262144, 9))
    features_37_weight = np.zeros((262144, 4))
    features_37_bias = np.zeros((256,))
    features_38_weight = np.zeros((256,))
    features_38_bias = np.zeros((256,))
    # 48
    # features_40_weight = np.zeros((262144, 9))
    features_40_weight = np.zeros((262144, 4))
    features_40_bias = np.zeros((256,))
    features_41_weight = np.zeros((256,))
    features_41_bias = np.zeros((256,))

    classifier_weight = np.zeros((10, 256))
    classifier_bias = np.zeros((5,))
    return features_0_weight, features_0_bias, features_1_weight, features_1_bias, features_3_weight, features_3_bias, \
           features_4_weight, features_4_bias, features_7_weight, features_7_bias, features_8_weight, features_8_bias, \
           features_10_weight, features_10_bias, features_11_weight, features_11_bias, features_14_weight, features_14_bias, \
           features_15_weight, features_15_bias, features_17_weight, features_17_bias, features_18_weight, features_18_bias, \
           features_20_weight, features_20_bias, features_21_weight, features_21_bias, features_24_weight, features_24_bias, \
           features_25_weight, features_25_bias, features_27_weight, features_27_bias, features_28_weight, features_28_bias, \
           features_30_weight, features_30_bias, features_31_weight, features_31_bias, features_34_weight, features_34_bias, \
           features_35_weight, features_35_bias, features_37_weight, features_37_bias, features_38_weight, features_38_bias, \
           features_40_weight, features_40_bias, features_41_weight, features_41_bias, classifier_weight, classifier_bias

def init_cnn_alpha_r():
    # 0
    # features_0_weight = np.zeros((192, 9))
    features_0_weight = np.zeros((192, 5))
    features_0_bias = np.zeros((32,))
    features_1_weight = np.zeros((32,))
    features_1_bias = np.zeros((32,))
    # 4
    # features_3_weight = np.zeros((, 9))
    features_3_weight = np.zeros((4096, 5))
    features_3_bias = np.zeros((32,))
    features_4_weight = np.zeros((32,))
    features_4_bias = np.zeros((32,))
    # 8
    # features_7_weight = np.zeros((, 9))
    features_7_weight = np.zeros((8192, 5))
    features_7_bias = np.zeros((64,))
    features_8_weight = np.zeros((64,))
    features_8_bias = np.zeros((64,))
    # 12
    # features_10_weight = np.zeros((16384, 9))
    features_10_weight = np.zeros((16384, 5))
    features_10_bias = np.zeros((64,))
    features_11_weight = np.zeros((64,))
    features_11_bias = np.zeros((64,))
    # 16
    # features_14_weight = np.zeros((32768, 9))
    features_14_weight = np.zeros((32768, 5))
    features_14_bias = np.zeros((128,))
    features_15_weight = np.zeros((128,))
    features_15_bias = np.zeros((128,))
    # 20
    # features_17_weight = np.zeros((65536, 9))
    features_17_weight = np.zeros((65536, 5))
    features_17_bias = np.zeros((128,))
    features_18_weight = np.zeros((128,))
    features_18_bias = np.zeros((128,))
    # 24
    # features_20_weight = np.zeros((65536, 9))
    features_20_weight = np.zeros((65536, 5))
    features_20_bias = np.zeros((128,))
    features_21_weight = np.zeros((128,))
    features_21_bias = np.zeros((128,))
    # 28
    # features_24_weight = np.zeros((131072, 9))
    features_24_weight = np.zeros((131072, 5))
    features_24_bias = np.zeros((256,))
    features_25_weight = np.zeros((256,))
    features_25_bias = np.zeros((256,))
    # 32
    # features_27_weight = np.zeros((262144, 9))
    features_27_weight = np.zeros((262144, 5))
    features_27_bias = np.zeros((256,))
    features_28_weight = np.zeros((256,))
    features_28_bias = np.zeros((256,))
    # 36
    # features_30_weight = np.zeros((262144, 9))
    features_30_weight = np.zeros((262144, 5))
    features_30_bias = np.zeros((256,))
    features_31_weight = np.zeros((256,))
    features_31_bias = np.zeros((256,))
    # 40
    # features_34_weight = np.zeros((262144, 9))
    features_34_weight = np.zeros((262144, 5))
    features_34_bias = np.zeros((256,))
    features_35_weight = np.zeros((256,))
    features_35_bias = np.zeros((256,))
    # 44
    # features_37_weight = np.zeros((262144, 9))
    features_37_weight = np.zeros((262144, 5))
    features_37_bias = np.zeros((256,))
    features_38_weight = np.zeros((256,))
    features_38_bias = np.zeros((256,))
    # 48
    # features_40_weight = np.zeros((262144, 9))
    features_40_weight = np.zeros((262144, 5))
    features_40_bias = np.zeros((256,))
    features_41_weight = np.zeros((256,))
    features_41_bias = np.zeros((256,))

    classifier_weight = np.zeros((10, 256))
    classifier_bias = np.zeros((5,))
    return features_0_weight, features_0_bias, features_1_weight, features_1_bias, features_3_weight, features_3_bias, \
           features_4_weight, features_4_bias, features_7_weight, features_7_bias, features_8_weight, features_8_bias, \
           features_10_weight, features_10_bias, features_11_weight, features_11_bias, features_14_weight, features_14_bias, \
           features_15_weight, features_15_bias, features_17_weight, features_17_bias, features_18_weight, features_18_bias, \
           features_20_weight, features_20_bias, features_21_weight, features_21_bias, features_24_weight, features_24_bias, \
           features_25_weight, features_25_bias, features_27_weight, features_27_bias, features_28_weight, features_28_bias, \
           features_30_weight, features_30_bias, features_31_weight, features_31_bias, features_34_weight, features_34_bias, \
           features_35_weight, features_35_bias, features_37_weight, features_37_bias, features_38_weight, features_38_bias, \
           features_40_weight, features_40_bias, features_41_weight, features_41_bias, classifier_weight, classifier_bias

def init_cnn_beta():
    # 0
    # features_0_weight = np.zeros((192, 9))
    features_0_weight = np.zeros((192, 5))
    features_0_bias = np.zeros((32,))
    features_1_weight = np.zeros((32,))
    features_1_bias = np.zeros((32,))
    # 4
    # features_3_weight = np.zeros((, 9))
    features_3_weight = np.zeros((4096, 5))
    features_3_bias = np.zeros((32,))
    features_4_weight = np.zeros((32,))
    features_4_bias = np.zeros((32,))
    # 8
    # features_7_weight = np.zeros((, 9))
    features_7_weight = np.zeros((8192, 5))
    features_7_bias = np.zeros((64,))
    features_8_weight = np.zeros((64,))
    features_8_bias = np.zeros((64,))
    # 12
    # features_10_weight = np.zeros((16384, 9))
    features_10_weight = np.zeros((16384, 5))
    features_10_bias = np.zeros((64,))
    features_11_weight = np.zeros((64,))
    features_11_bias = np.zeros((64,))
    # 16
    # features_14_weight = np.zeros((32768, 9))
    features_14_weight = np.zeros((32768, 5))
    features_14_bias = np.zeros((128,))
    features_15_weight = np.zeros((128,))
    features_15_bias = np.zeros((128,))
    # 20
    # features_17_weight = np.zeros((65536, 9))
    features_17_weight = np.zeros((65536, 5))
    features_17_bias = np.zeros((128,))
    features_18_weight = np.zeros((128,))
    features_18_bias = np.zeros((128,))
    # 24
    # features_20_weight = np.zeros((65536, 9))
    features_20_weight = np.zeros((65536, 5))
    features_20_bias = np.zeros((128,))
    features_21_weight = np.zeros((128,))
    features_21_bias = np.zeros((128,))
    # 28
    # features_24_weight = np.zeros((131072, 9))
    features_24_weight = np.zeros((131072, 5))
    features_24_bias = np.zeros((256,))
    features_25_weight = np.zeros((256,))
    features_25_bias = np.zeros((256,))
    # 32
    # features_27_weight = np.zeros((262144, 9))
    features_27_weight = np.zeros((262144, 5))
    features_27_bias = np.zeros((256,))
    features_28_weight = np.zeros((256,))
    features_28_bias = np.zeros((256,))
    # 36
    # features_30_weight = np.zeros((262144, 9))
    features_30_weight = np.zeros((262144, 5))
    features_30_bias = np.zeros((256,))
    features_31_weight = np.zeros((256,))
    features_31_bias = np.zeros((256,))
    # 40
    # features_34_weight = np.zeros((262144, 9))
    features_34_weight = np.zeros((262144, 5))
    features_34_bias = np.zeros((256,))
    features_35_weight = np.zeros((256,))
    features_35_bias = np.zeros((256,))
    # 44
    # features_37_weight = np.zeros((262144, 9))
    features_37_weight = np.zeros((262144, 5))
    features_37_bias = np.zeros((256,))
    features_38_weight = np.zeros((256,))
    features_38_bias = np.zeros((256,))
    # 48
    # features_40_weight = np.zeros((262144, 9))
    features_40_weight = np.zeros((262144, 5))
    features_40_bias = np.zeros((256,))
    features_41_weight = np.zeros((256,))
    features_41_bias = np.zeros((256,))

    classifier_weight = np.zeros((10, 256))
    classifier_bias = np.zeros((5,))
    return features_0_weight, features_0_bias, features_1_weight, features_1_bias, features_3_weight, features_3_bias, \
           features_4_weight, features_4_bias, features_7_weight, features_7_bias, features_8_weight, features_8_bias, \
           features_10_weight, features_10_bias, features_11_weight, features_11_bias, features_14_weight, features_14_bias, \
           features_15_weight, features_15_bias, features_17_weight, features_17_bias, features_18_weight, features_18_bias, \
           features_20_weight, features_20_bias, features_21_weight, features_21_bias, features_24_weight, features_24_bias, \
           features_25_weight, features_25_bias, features_27_weight, features_27_bias, features_28_weight, features_28_bias, \
           features_30_weight, features_30_bias, features_31_weight, features_31_bias, features_34_weight, features_34_bias, \
           features_35_weight, features_35_bias, features_37_weight, features_37_bias, features_38_weight, features_38_bias, \
           features_40_weight, features_40_bias, features_41_weight, features_41_bias, classifier_weight, classifier_bias

def init_cnn_beta_r():
    # 0
    # features_0_weight = np.zeros((192, 9))
    features_0_weight = np.zeros((192, 4))
    features_0_bias = np.zeros((32,))
    features_1_weight = np.zeros((32,))
    features_1_bias = np.zeros((32,))
    # 4
    # features_3_weight = np.zeros((, 9))
    features_3_weight = np.zeros((4096, 4))
    features_3_bias = np.zeros((32,))
    features_4_weight = np.zeros((32,))
    features_4_bias = np.zeros((32,))
    # 8
    # features_7_weight = np.zeros((, 9))
    features_7_weight = np.zeros((8192, 4))
    features_7_bias = np.zeros((64,))
    features_8_weight = np.zeros((64,))
    features_8_bias = np.zeros((64,))
    # 12
    # features_10_weight = np.zeros((16384, 9))
    features_10_weight = np.zeros((16384, 4))
    features_10_bias = np.zeros((64,))
    features_11_weight = np.zeros((64,))
    features_11_bias = np.zeros((64,))
    # 16
    # features_14_weight = np.zeros((32768, 9))
    features_14_weight = np.zeros((32768, 4))
    features_14_bias = np.zeros((128,))
    features_15_weight = np.zeros((128,))
    features_15_bias = np.zeros((128,))
    # 20
    # features_17_weight = np.zeros((65536, 9))
    features_17_weight = np.zeros((65536, 4))
    features_17_bias = np.zeros((128,))
    features_18_weight = np.zeros((128,))
    features_18_bias = np.zeros((128,))
    # 24
    # features_20_weight = np.zeros((65536, 9))
    features_20_weight = np.zeros((65536, 4))
    features_20_bias = np.zeros((128,))
    features_21_weight = np.zeros((128,))
    features_21_bias = np.zeros((128,))
    # 28
    # features_24_weight = np.zeros((131072, 9))
    features_24_weight = np.zeros((131072, 4))
    features_24_bias = np.zeros((256,))
    features_25_weight = np.zeros((256,))
    features_25_bias = np.zeros((256,))
    # 32
    # features_27_weight = np.zeros((262144, 9))
    features_27_weight = np.zeros((262144, 4))
    features_27_bias = np.zeros((256,))
    features_28_weight = np.zeros((256,))
    features_28_bias = np.zeros((256,))
    # 36
    # features_30_weight = np.zeros((262144, 9))
    features_30_weight = np.zeros((262144, 4))
    features_30_bias = np.zeros((256,))
    features_31_weight = np.zeros((256,))
    features_31_bias = np.zeros((256,))
    # 40
    # features_34_weight = np.zeros((262144, 9))
    features_34_weight = np.zeros((262144, 4))
    features_34_bias = np.zeros((256,))
    features_35_weight = np.zeros((256,))
    features_35_bias = np.zeros((256,))
    # 44
    # features_37_weight = np.zeros((262144, 9))
    features_37_weight = np.zeros((262144, 4))
    features_37_bias = np.zeros((256,))
    features_38_weight = np.zeros((256,))
    features_38_bias = np.zeros((256,))
    # 48
    # features_40_weight = np.zeros((262144, 9))
    features_40_weight = np.zeros((262144, 4))
    features_40_bias = np.zeros((256,))
    features_41_weight = np.zeros((256,))
    features_41_bias = np.zeros((256,))

    classifier_weight = np.zeros((10, 256))
    classifier_bias = np.zeros((5,))
    return features_0_weight, features_0_bias, features_1_weight, features_1_bias, features_3_weight, features_3_bias, \
           features_4_weight, features_4_bias, features_7_weight, features_7_bias, features_8_weight, features_8_bias, \
           features_10_weight, features_10_bias, features_11_weight, features_11_bias, features_14_weight, features_14_bias, \
           features_15_weight, features_15_bias, features_17_weight, features_17_bias, features_18_weight, features_18_bias, \
           features_20_weight, features_20_bias, features_21_weight, features_21_bias, features_24_weight, features_24_bias, \
           features_25_weight, features_25_bias, features_27_weight, features_27_bias, features_28_weight, features_28_bias, \
           features_30_weight, features_30_bias, features_31_weight, features_31_bias, features_34_weight, features_34_bias, \
           features_35_weight, features_35_bias, features_37_weight, features_37_bias, features_38_weight, features_38_bias, \
           features_40_weight, features_40_bias, features_41_weight, features_41_bias, classifier_weight, classifier_bias

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
