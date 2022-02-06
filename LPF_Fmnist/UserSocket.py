import random
import socket
import sys
import time

import numpy as np
import json

import torch

from LocalTrain import LocalTrain
from Initial import NpEncoder, para, port_user_alpha, port_user_beta, DataSize, para_dynamic, Logger, \
    dynamic_number, dynamic_iod
from Initial import gen_secret, gen_secret_join
from Initial import get_ip
from Initial import Net
from Mask import secret_seed
from Hash import SHA256

sys.stdout = Logger('log_user.txt')

user_number, total_round = para()
user_number_base = user_number
dynamic, dynamic_range = para_dynamic()
dynamic_seed = 10
# r, c = dimension()
buff_size = 50000000

# Each client creates a random seed corresponding to the server
Addr = get_ip()
GlobalModel = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = 109
g = 6
public_a = 77
public_b = 47
private_u = 13


def main():
    global user_number, user_join, user_quit, dynamic_seed, user_number_m

    seed_0 = gen_secret(user_number, private_u, public_a)
    seed_1 = gen_secret(user_number, private_u, public_b)

    if dynamic == 1:
        user_number_t = user_number + (user_number_base * dynamic_range // 100)
        user_number_m = user_number - (user_number_base * dynamic_range // 100)
    else:
        user_number_t = user_number

    user0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    user0.connect((Addr, port_user_alpha()))

    user1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    user1.connect((Addr, port_user_beta()))

    # user0.send(str(c).encode('utf-8'))
    # user1.send(str(c).encode('utf-8'))

    DataSetSize = DataSize.size_of_data(user_number_t)
    sendSize = json.dumps(DataSetSize)
    user0.send(sendSize.encode('utf-8'))
    user1.send(sendSize.encode('utf-8'))

    time_total_pm = 0
    t_acc = np.zeros((total_round,))

    for e in range(total_round):
        print('*****************')
        print('Start round %d' % int(e + 1))
        print('*****************')
        iod = dynamic_iod(dynamic_seed)

        if e > 0:
            if dynamic == 1:
                if iod == 1:
                    if user_number < user_number_t:
                        tmp = user_number
                        user_number = user_number + dynamic_number(dynamic_seed + 100, user_number_t, user_number)
                        for k in range(user_number - tmp):
                            seed_0.append(gen_secret_join(tmp + k, private_u, public_a))
                            seed_1.append(gen_secret_join(tmp + k, private_u, public_b))
                if iod == 0:
                    if user_number > 2:
                        if user_number > user_number_m:
                            tmp = user_number
                            user_number = user_number - dynamic_number(dynamic_seed, user_number_m, user_number)
                            tmpp = tmp - user_number
                            for k in range(tmpp):  # delete the quit users
                                del seed_0[tmp - 1 - k]
                                del seed_1[tmp - 1 - k]
                                tmp -= 1
        print('The number of users in round %d: %d' % (int(e + 1), user_number))
        time_apm = 0
        decode = {}
        acc = np.zeros((user_number,))
        a_acc = 0.0

        for j in range(user_number):
            tLen_0 = 0
            tLen_1 = 0
            time_pm = 0

            # Local Train
            data, acc[j] = LocalTrain(GlobalModel, j)
            print('user%d--acc--%f' % (j + 1, acc[j]))
            a_acc += acc[j]
            # sendData = json.dumps(data, cls=NpEncoder)
            if j == 0:
                print("users start training")
                # print('The size of a local gradient：%d KB' % (len(sendData) / 1024))

            for i in range(8):
                # Partition and Masking
                time0 = time.perf_counter()
                if i % 2 == 0:
                    data_split_0 = data[i][:, : (data[i].shape[1] // 2)]
                    data_split_1 = data[i][:, (data[i].shape[1] // 2):]
                    data_split_0 = data_split_0 + secret_seed(seed_1[j], data_split_0.shape[0], data_split_0.shape[1])
                    data_split_1 = data_split_1 + secret_seed(seed_0[j], data_split_1.shape[0], data_split_1.shape[1])
                else:
                    data_split_0 = data[i][: (data[i].shape[0] // 2), ]
                    data_split_1 = data[i][(data[i].shape[0] // 2):, ]
                    data_split_0 = data_split_0 + secret_seed(seed_1[j], data_split_0.shape[0], 0)
                    data_split_1 = data_split_1 + secret_seed(seed_0[j], data_split_1.shape[0], 0)
                time1 = time.perf_counter()
                time_pm += (time1 - time0)
                sendData_0 = json.dumps(data_split_0, cls=NpEncoder)
                sendData_1 = json.dumps(data_split_1, cls=NpEncoder)
                user0.send(sendData_0.encode('utf-8'))
                user1.send(sendData_1.encode('utf-8'))
                time.sleep(0.1)

                tLen_0 += (len(sendData_0) / 1024)
                tLen_1 += (len(sendData_1) / 1024)

            times0 = time.perf_counter()
            sha256 = SHA256()  # Update secrets
            seed_0[j] = (int(sha256.hash(str(seed_0[j])), 16) // 10 ** 72)
            seed_1[j] = (int(sha256.hash(str(seed_1[j])), 16) // 10 ** 72)
            times1 = time.perf_counter()

            time_apm += (time_pm + times1 - times0)
            if j == 0:
                print('The local gradient sent to the S_alpha--%dKB--' % tLen_0)
                print('The local gradient sent to the S_beta--%dKB--' % tLen_1)
                print("-----")

        print('Global accuracy---%f' % (a_acc / user_number))
        t_acc[e] += (a_acc / user_number)
        print('the cryptographic overhead of users in round %d：%f ms' % (e + 1, (time_apm * 1000) / user_number))
        print('-----')
        time_total_pm += (time_apm * 1000) / user_number

        for i in range(8):
            recvData = np.array(json.loads(user1.recv(int(buff_size))))
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
        dynamic_seed += 6

    print('+++++++++++')
    print('Users, the cryptographic overhead in all rounds：%f ms' % (time_total_pm / total_round))
    print('+++++++++++')
    print('Accs:')
    print(t_acc)
    print('+++++++++++')
    user0.close()
    user1.close()


if __name__ == '__main__':
    main()
