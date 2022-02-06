import socket
import sys
import time

import numpy as np
import json

from Initial import para, port_user_alpha, Logger, init_cnn_alpha
from Initial import NpEncoder
from Initial import get_ip

sys.stdout = Logger('log_server.txt')

Addr = get_ip()
user_number, total_round = para()
buff_size = 100000000
tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpSerSock.bind(("", port_user_alpha()))
tcpSerSock.listen(5)
round = 0


tcpCliSock, addr = tcpSerSock.accept()
print('S connect to users:', addr)
time_total_avg = 0
time_total_recover = 0

# c = tcpCliSock.recv(int(buff_size))
# c = int(json.loads(c))
size = []
total_size = 0
recvSize = tcpCliSock.recv(65535)
DataSetSize = json.loads(recvSize)
for i in range(user_number):
    if i < user_number:
        total_size += DataSetSize[i]
    size.append(DataSetSize[i])


while True:
    decode = {}
    recover = {}

    fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias, fc4_weight, fc4_bias = init_cnn_alpha()
    time_avg = 0
    for i in range(user_number):
        decode = {}
        for j in range(8):
            data = tcpCliSock.recv(int(buff_size))
            data = np.array(json.loads(data))

            decode.setdefault(j, data)
        fc1_weight += decode[0] * (size[i] / total_size)
        fc1_bias += decode[1] * (size[i] / total_size)
        fc2_weight += decode[2] * (size[i] / total_size)
        fc2_bias += decode[3] * (size[i] / total_size)
        fc3_weight += decode[4] * (size[i] / total_size)
        fc3_bias += decode[5] * (size[i] / total_size)
        fc4_weight += decode[6] * (size[i] / total_size)
        fc4_bias += decode[7] * (size[i] / total_size)

    sendData = {0: fc1_weight, 1: fc1_bias, 2: fc2_weight, 3: fc2_bias, 4: fc3_weight, 5: fc3_bias,
                6: fc4_weight, 7: fc4_bias}

    tLen = 0
    for i in range(8):
        send_result = json.dumps(sendData[i], cls=NpEncoder)
        tcpCliSock.send(send_result.encode('utf-8'))
        tLen += (len(send_result) / 1024)
        time.sleep(0.5)
    print('S sends the %d KB aggregation gradient to the S_beta' % tLen)
    round += 1
    if round == total_round:
        break
print('+++++++++++')
print('Server, The average time of federal average in all rounds：%f ms' % (time_total_avg / total_round))
print('+++++++++++')
time.sleep(3)
tcpCliSock.close()
tcpSerSock.close()
