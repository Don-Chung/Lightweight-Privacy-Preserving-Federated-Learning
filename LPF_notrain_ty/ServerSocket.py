import socket
import sys
import time

import numpy as np
import json

from Initial import para, port_user_alpha, Logger
from Initial import NpEncoder
from Initial import get_ip

sys.stdout = Logger('log_server.txt')

Addr = get_ip()
user_number, total_round = para()
buff_size = 6553500
tcpSerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpSerSock.bind(("", port_user_alpha()))
tcpSerSock.listen(5)
round = 0


tcpCliSock, addr = tcpSerSock.accept()
print('S_alpha connect to users:', addr)
time_total_avg = 0
time_total_recover = 0

# c = tcpCliSock.recv(int(buff_size))
# c = int(json.loads(c))

while True:
    decode = {}
    recover = {}

    np.random.seed(100)  # Simulate user data set size
    size = np.random.randint(1, user_number, (user_number,))
    total_size = 0
    time_avg = 0
    for i in range(user_number):
        data = tcpCliSock.recv(int(buff_size))
        data = np.array(json.loads(data))
        total_size = total_size + size[i]
        decode.setdefault(i, data)

    Favg = np.zeros(decode[0].shape)

    # Federal average
    time0 = time.perf_counter()
    for i in range(user_number):
        Favg = Favg + (decode[i] * (size[i] / total_size))
    time1 = time.perf_counter()
    time_avg += (time1 - time0)

    print('Average federal time of Server in round %d：%f ms' % (round + 1, time_avg * 1000))
    print('-----')
    time_total_avg += time_avg * 1000

    tLen = 0
    send_result = json.dumps(Favg, cls=NpEncoder)
    tcpCliSock.send(send_result.encode('utf-8'))
    tLen += (len(send_result)/1024)
    print('Server sends the %d KB aggregation gradient to the user' % tLen)
    round += 1
    if round == total_round:
        break
print('+++++++++++')
print('Server, The average time of federal average in all rounds：%f ms' % (time_total_avg / total_round))
print('+++++++++++')
time.sleep(3)
tcpCliSock.close()
tcpSerSock.close()
