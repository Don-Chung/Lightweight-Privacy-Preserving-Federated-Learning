import socket
import sys
import time

import numpy as np
import json

import torch

from LocalTrain import LocalTrain
from Initial import NpEncoder, para, port_user_alpha, Logger, Net, DataSize, VGG16
from Initial import get_ip

sys.stdout = Logger('log_user.txt')

user_number, total_round = para()
buff_size = 1000000000

Addr = get_ip()
GlobalModel = VGG16()
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
        for i in range(54):
            # print('--%d--' % i)
            sendData = json.dumps(data[i], cls=NpEncoder)
            user0.send(sendData.encode('utf-8'))
            time.sleep(2)
            tLen += (len(sendData) / 1024)

        if j == 0:
            print('The local gradient sent to the S--%dKB--' % tLen)
            print("-----")
    print('Global accuracy---%f' % (a_acc / user_number))
    t_acc[e] += (a_acc / user_number)
    for i in range(54):
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
    model_dict['features.0.weight'] += torch.from_numpy(decode[0])
    model_dict['features.0.bias'] += torch.from_numpy(decode[1])
    model_dict['features.1.weight'] += torch.from_numpy(decode[2])
    model_dict['features.1.bias'] += torch.from_numpy(decode[3])
    model_dict['features.3.weight'] += torch.from_numpy(decode[4])
    model_dict['features.3.bias'] += torch.from_numpy(decode[5])
    model_dict['features.4.weight'] += torch.from_numpy(decode[6])
    model_dict['features.4.bias'] += torch.from_numpy(decode[7])

    model_dict['features.7.weight'] += torch.from_numpy(decode[8])
    model_dict['features.7.bias'] += torch.from_numpy(decode[9])
    model_dict['features.8.weight'] += torch.from_numpy(decode[10])
    model_dict['features.8.bias'] += torch.from_numpy(decode[11])
    model_dict['features.10.weight'] += torch.from_numpy(decode[12])
    model_dict['features.10.bias'] += torch.from_numpy(decode[13])
    model_dict['features.11.weight'] += torch.from_numpy(decode[14])
    model_dict['features.11.bias'] += torch.from_numpy(decode[15])

    model_dict['features.14.weight'] += torch.from_numpy(decode[16])
    model_dict['features.14.bias'] += torch.from_numpy(decode[17])
    model_dict['features.15.weight'] += torch.from_numpy(decode[18])
    model_dict['features.15.bias'] += torch.from_numpy(decode[19])
    model_dict['features.17.weight'] += torch.from_numpy(decode[20])
    model_dict['features.17.bias'] += torch.from_numpy(decode[21])
    model_dict['features.18.weight'] += torch.from_numpy(decode[22])
    model_dict['features.18.bias'] += torch.from_numpy(decode[23])

    model_dict['features.20.weight'] += torch.from_numpy(decode[24])
    model_dict['features.20.bias'] += torch.from_numpy(decode[25])
    model_dict['features.21.weight'] += torch.from_numpy(decode[26])
    model_dict['features.21.bias'] += torch.from_numpy(decode[27])
    model_dict['features.24.weight'] += torch.from_numpy(decode[28])
    model_dict['features.24.bias'] += torch.from_numpy(decode[29])
    model_dict['features.25.weight'] += torch.from_numpy(decode[30])
    model_dict['features.25.bias'] += torch.from_numpy(decode[31])

    model_dict['features.27.weight'] += torch.from_numpy(decode[32])
    model_dict['features.27.bias'] += torch.from_numpy(decode[33])
    model_dict['features.28.weight'] += torch.from_numpy(decode[34])
    model_dict['features.28.bias'] += torch.from_numpy(decode[35])
    model_dict['features.30.weight'] += torch.from_numpy(decode[36])
    model_dict['features.30.bias'] += torch.from_numpy(decode[37])
    model_dict['features.31.weight'] += torch.from_numpy(decode[38])
    model_dict['features.31.bias'] += torch.from_numpy(decode[39])

    model_dict['features.34.weight'] += torch.from_numpy(decode[40])
    model_dict['features.34.bias'] += torch.from_numpy(decode[41])
    model_dict['features.35.weight'] += torch.from_numpy(decode[42])
    model_dict['features.35.bias'] += torch.from_numpy(decode[43])
    model_dict['features.37.weight'] += torch.from_numpy(decode[44])
    model_dict['features.37.bias'] += torch.from_numpy(decode[45])
    model_dict['features.38.weight'] += torch.from_numpy(decode[46])
    model_dict['features.38.bias'] += torch.from_numpy(decode[47])

    model_dict['features.40.weight'] += torch.from_numpy(decode[48])
    model_dict['features.40.bias'] += torch.from_numpy(decode[49])
    model_dict['features.41.weight'] += torch.from_numpy(decode[50])
    model_dict['features.41.bias'] += torch.from_numpy(decode[51])
    model_dict['classifier.weight'] += torch.from_numpy(decode[52])
    model_dict['classifier.bias'] += torch.from_numpy(decode[53])

    GlobalModel.load_state_dict(model_dict)

print('+++++++++++')
print('Accs:')
print(t_acc)
print('+++++++++++')
user0.close()
