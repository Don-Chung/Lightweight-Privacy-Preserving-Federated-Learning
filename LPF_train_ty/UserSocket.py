import socket
import sys
import time

import numpy as np
import json

import torch

from LocalTrain import train, LocalTrain
from Initial import NpEncoder, para, dimension, port_user_alpha, Logger, Net, DataSize
from Initial import get_ip

sys.stdout = Logger('log_user.txt')

user_number, total_round = para()
buff_size = 100000000

Addr = get_ip()
GlobalModel = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
user0.connect((Addr, port_user_alpha()))

t_acc = np.zeros((total_round,))

DataSetSize = DataSize.size_of_data(user_number)
sendSize = json.dumps(DataSetSize)
user0.send(sendSize.encode('utf-8'))

for e in range(total_round):
    print('*****************')
    print('Start round %d' % int(e + 1))
    print('*****************')
    decode = {}
    acc = np.zeros((user_number,))
    a_acc = 0.0

    for j in range(user_number):
        tLen = 0
        data, acc[j] = LocalTrain(GlobalModel, j)
        print('user%d--acc--%f' % (j + 1, acc[j]))
        a_acc += acc[j]
        if j == 0:
            print("users start training")
        for i in range(8):
            sendData = json.dumps(data[i], cls=NpEncoder)
            user0.send(sendData.encode('utf-8'))
            time.sleep(0.5)
            tLen += (len(sendData) / 1024)

        if j == 0:
            print('The local gradient sent to the S--%dKB--' % tLen)
            print("-----")
    print('Global accuracy---%f' % (a_acc / user_number))
    t_acc[e] += (a_acc / user_number)
    for i in range(8):
        recvData = np.array(json.loads(user0.recv(int(buff_size))))
        decode.setdefault(i, recvData)  # 存放解码后的数据
    print('Global model received')

    for i in GlobalModel.state_dict():
        if device == 'cpu':
            GlobalModel.state_dict()[i] -= GlobalModel.to(device).state_dict()[i]
        else:
            GlobalModel.state_dict()[i] -= GlobalModel.state_dict()[i]
    torch.save(GlobalModel.state_dict(),
               "./model_state_dict_" + ".pt")
    model_dict = torch.load("model_state_dict_" + ".pt")
    model_dict['fc1.weight'] += torch.from_numpy(decode[0])
    model_dict['fc1.bias'] += torch.from_numpy(decode[1])
    model_dict['fc2.weight'] += torch.from_numpy(decode[2])
    model_dict['fc2.bias'] += torch.from_numpy(decode[3])
    model_dict['fc3.weight'] += torch.from_numpy(decode[4])
    model_dict['fc3.bias'] += torch.from_numpy(decode[5])
    model_dict['fc4.weight'] += torch.from_numpy(decode[6])
    model_dict['fc4.bias'] += torch.from_numpy(decode[7])

    GlobalModel.load_state_dict(model_dict)

print('+++++++++++')
print('Accs:')
print(t_acc)
print('+++++++++++')
user0.close()
