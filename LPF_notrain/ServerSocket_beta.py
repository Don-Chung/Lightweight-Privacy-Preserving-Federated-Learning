import socket
import sys
import time

import numpy as np
import json

from Hash import SHA256
from Initial import gen_secret, para, port_user_beta, port_alpha_beta, Logger
from Initial import NpEncoder

from mask import secret_seed

sys.stdout = Logger('log_beta.txt')

p = 109
g = 6
public_u = 96
private_b = 17

user_number, total_round = para()
buff_size = 6553500
tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpSerSock.bind(("", port_user_beta()))
tcpSerSock.listen(5)
SSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SSerSock.bind(("", port_alpha_beta()))
SSerSock.listen(5)
round = 0

# Generate the seed corresponding to the user
seed_1 = gen_secret(user_number, private_b, public_u)

tcpCliSock, addr = tcpSerSock.accept()
print('S_beta connect to users:', addr)
SCliSock, addrS = SSerSock.accept()
print('S_beta connect to S_alpha:', addrS)
time_total_avg = 0
time_total_recover = 0

c = tcpCliSock.recv(int(buff_size))
c = int(json.loads(c))

while True:
    decode = {}
    np.random.seed(100)  # Simulate user data set size
    size = np.random.randint(1, user_number, (user_number,))
    total_size = 0
    time_avg = 0
    time_recover = 0
    for i in range(user_number):
        data = tcpCliSock.recv(int(buff_size))
        data = np.array(json.loads(data))
        total_size = total_size + size[i]
        decode.setdefault(i, data)

    Favg = np.zeros(decode[0].shape)
    Frecover = np.zeros(decode[0].shape)

    # Federal average
    time0 = time.perf_counter()
    for i in range(user_number):
        Favg = Favg + (decode[i] * (size[i] / total_size))
        recover = - secret_seed(seed_1[i], decode[i].shape[0], c - decode[i].shape[1])
        Frecover = Frecover + (recover * (size[i] / total_size))
    time1 = time.perf_counter()
    time_avg += (time1 - time0)

    recvData_0 = np.array(json.loads(SCliSock.recv(int(buff_size))))

    # Recovery and De-masking
    time0 = time.perf_counter()
    sendData = np.hstack((Favg, Frecover))
    sendData += recvData_0
    for i in range(user_number):
        sha256 = SHA256()  # Update seeds
        seed_1[i] = (int(sha256.hash(str(seed_1[i])), 16) // 10 ** 72)
    time1 = time.perf_counter()
    time_recover += (time1 - time0)

    print('Average federal time of S_beta in round %d：%f ms' % (round + 1, time_avg * 1000))
    print('the cryptographic overhead of S_beta in round %d：%f ms' % (round + 1, time_recover * 1000))
    print('the computation overhead of S_beta in round %d：%f ms' % (round + 1, (time_avg + time_recover) * 1000))
    print('-----')
    time_total_avg += time_avg * 1000
    time_total_recover += time_recover * 1000

    tLen = 0
    send_result = json.dumps(sendData, cls=NpEncoder)
    tcpCliSock.send(send_result.encode('utf-8'))
    tLen += (len(send_result) / 1024)
    print('S_beta sent %dKB global gradient to each user' % tLen)
    round += 1
    if round == total_round:
        break
print('+++++++++++')
print('S_beta, The average time of federal average in all rounds：%f ms' % (time_total_avg / total_round))
print('S_beta, The average time of the cryptographic overhead in all rounds：%f ms' % (time_total_recover / total_round))
print('S_beta, The average time of the computation overhead in all rounds：%f ms' % (
            (time_total_recover + time_total_avg) / total_round))
print('+++++++++++')
SCliSock.close()
tcpCliSock.close()
SSerSock.close()
tcpSerSock.close()
