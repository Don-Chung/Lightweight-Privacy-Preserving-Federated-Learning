import socket
import sys
import time

import numpy as np
import json
from LocalTrain import train
from Initial import NpEncoder, para, dimension, port_user_alpha, Logger
from Initial import get_ip

sys.stdout = Logger('log_user.txt')

user_number, total_round = para()
r, c = dimension()
buff_size = 6553500

Addr = get_ip()

user0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
user0.connect((Addr, port_user_alpha()))

user0.send(str(c).encode('utf-8'))

for e in range(total_round):
    print('*****************')
    print('Start round %d' % int(e + 1))
    print('*****************')

    # Local Train
    data = train(user_number, r, c)

    for j in range(user_number):
        sendData = json.dumps(data[j], cls=NpEncoder)
        if j == 0:
            print("users starts training")
            print('The size of the local gradient：%d KB' % (len(sendData) / 1024))

        user0.send(sendData.encode('utf-8'))
        time.sleep(0.1)

    recvData = np.array(json.loads(user0.recv(int(buff_size))))

print('+++++++++++')
print('+++++++++++')
user0.close()
