import random
import socket
import sys
import time

import numpy as np
import json

from Hash import SHA256
from Initial import gen_secret, para, port_user_alpha, port_alpha_beta, init_cnn_alpha, para_dynamic, Logger, \
    gen_secret_join, dynamic_number, dynamic_iod
from Initial import NpEncoder
from Initial import get_ip

from Mask import secret_seed

sys.stdout = Logger('log_alpha.txt')

Addr = get_ip()
user_number, total_round = para()
user_number_base = user_number
global user_number_m
dynamic, dynamic_range = para_dynamic()
dynamic_seed = 10
# buff_size = 6553500
buff_size = 50000000
tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpSerSock.bind(("", port_user_alpha()))
tcpSerSock.listen(5)
StS = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
StS.connect((Addr, port_alpha_beta()))
round = 0
# c = [784, 1000, 1000, 500, 500, 200, 200, 10]

# Generate the seed corresponding to the user

p = 109
g = 6
public_u = 96
private_a = 15


tcpCliSock, addr = tcpSerSock.accept()
print('S_alpha connect to users:', addr)
time_total_avg = 0
time_total_recover = 0

if dynamic == 1:
    user_number_t = user_number + (user_number_base * dynamic_range // 100)
    user_number_m = user_number - (user_number_base * dynamic_range // 100)
else:
    user_number_t = user_number

seed_0 = gen_secret(user_number, private_a, public_u)

# c = tcpCliSock.recv(int(buff_size))
# c = int(json.loads(c))
size = []
total_size = 0
recvSize = tcpCliSock.recv(65535)
DataSetSize = json.loads(recvSize)
for i in range(user_number_t):
    if i < user_number:
        total_size += DataSetSize[i]
    size.append(DataSetSize[i])

while True:
    iod = dynamic_iod(dynamic_seed)

    if round > 0:
        if dynamic == 1:
            if iod == 1:
                if user_number < user_number_t:
                    tmp = user_number
                    user_number = user_number + dynamic_number(dynamic_seed + 100, user_number_t, user_number)
                    for k in range(user_number - tmp):
                        seed_0.append(gen_secret_join(tmp + k, private_a, public_u))
                        total_size += size[tmp + k]
            if iod == 0:
                if user_number > 2:
                    if user_number > user_number_m:
                        tmp = user_number
                        user_number = user_number - dynamic_number(dynamic_seed, user_number_m, user_number)
                        tmpp = tmp - user_number
                        for k in range(tmpp):  # delete the quit users
                            del seed_0[tmp - 1 - k]
                            total_size -= size[tmp - 1 - k]
    decode = {}
    recover = {}

    # np.random.seed(100)  # Simulate user data set size
    time_avg = 0
    time_recover = 0
    fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias, fc4_weight, fc4_bias = init_cnn_alpha()
    rc1_weight, rc1_bias, rc2_weight, rc2_bias, rc3_weight, rc3_bias, rc4_weight, rc4_bias = init_cnn_alpha()

    for i in range(user_number):
        decode = {}
        for j in range(8):
            data = tcpCliSock.recv(int(buff_size))
            data = np.array(json.loads(data))

            decode.setdefault(j, data)
        # Federal average
        time0 = time.perf_counter()
        fc1_weight += decode[0] * (size[i] / total_size)
        fc1_bias += decode[1] * (size[i] / total_size)
        fc2_weight += decode[2] * (size[i] / total_size)
        fc2_bias += decode[3] * (size[i] / total_size)
        fc3_weight += decode[4] * (size[i] / total_size)
        fc3_bias += decode[5] * (size[i] / total_size)
        fc4_weight += decode[6] * (size[i] / total_size)
        fc4_bias += decode[7] * (size[i] / total_size)

        rc1_weight += (- secret_seed(seed_0[i], decode[0].shape[0], decode[0].shape[1])) * (size[i] / total_size)
        rc1_bias += (- secret_seed(seed_0[i], decode[1].shape[0], 0)) * (size[i] / total_size)
        rc2_weight += (- secret_seed(seed_0[i], decode[2].shape[0], decode[2].shape[1])) * (size[i] / total_size)
        rc2_bias += (- secret_seed(seed_0[i], decode[3].shape[0], 0)) * (size[i] / total_size)
        rc3_weight += (- secret_seed(seed_0[i], decode[4].shape[0], decode[4].shape[1])) * (size[i] / total_size)
        rc3_bias += (- secret_seed(seed_0[i], decode[5].shape[0], 0)) * (size[i] / total_size)
        rc4_weight += (- secret_seed(seed_0[i], decode[6].shape[0], decode[6].shape[1])) * (size[i] / total_size)
        rc4_bias += (- secret_seed(seed_0[i], decode[7].shape[0], 0)) * (size[i] / total_size)

        time1 = time.perf_counter()
        time_avg += (time1 - time0)

    timef0 = time.perf_counter()
    fc1_weight = np.hstack((fc1_weight, rc1_weight))
    fc1_bias = np.hstack((fc1_bias, rc1_bias))
    fc2_weight = np.hstack((fc2_weight, rc2_weight))
    fc2_bias = np.hstack((fc2_bias, rc2_bias))
    fc3_weight = np.hstack((fc3_weight, rc3_weight))
    fc3_bias = np.hstack((fc3_bias, rc3_bias))
    fc4_weight = np.hstack((fc4_weight, rc4_weight))
    fc4_bias = np.hstack((fc4_bias, rc4_bias))
    timef1 = time.perf_counter()

    sendData = {0: fc1_weight, 1: fc1_bias, 2: fc2_weight, 3: fc2_bias, 4: fc3_weight, 5: fc3_bias,
                6: fc4_weight, 7: fc4_bias}
    time0 = time.perf_counter()
    for i in range(user_number):
        sha256 = SHA256()  # Update seeds
        seed_0[i] = (int(sha256.hash(str(seed_0[i])), 16) // 10 ** 72)
    time1 = time.perf_counter()
    time_recover += (time1 - time0 + timef1 - timef0)

    print('Average federal time of S_alpha in round %d：%f ms' % (round + 1, time_avg * 1000))
    print('the cryptographic overhead of S_alpha in round %d：%f ms' % (round + 1, time_recover * 1000))
    print('-----')
    time_total_avg += time_avg * 1000
    time_total_recover += time_recover * 1000

    tLen = 0
    for i in range(8):
        send_result = json.dumps(sendData[i], cls=NpEncoder)
        StS.send(send_result.encode('utf-8'))
        tLen += (len(send_result) / 1024)
        time.sleep(0.2)
    print('S_alpha sends the %d KB aggregation gradient to the S_beta' % tLen)
    dynamic_seed += 6
    round += 1
    if round == total_round:
        break
print('+++++++++++')
print('S_alpha, The average time of federal average in all rounds：%f ms' % (time_total_avg / total_round))
print('S_alpha, the cryptographic overhead in all rounds：%f ms' % (time_total_recover / total_round))
print('S_alpha, The average time of the computation overhead in all rounds：%f ms' % (
        (time_total_recover + time_total_avg) / total_round))
print('+++++++++++')
StS.close()
tcpCliSock.close()
tcpSerSock.close()
