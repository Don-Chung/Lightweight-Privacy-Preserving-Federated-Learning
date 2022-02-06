import socket
import sys
import time

import numpy as np
import json
from LocalTrain import train
from Initial import NpEncoder, para, dimension, port_user_alpha, port_user_beta, Logger
from Initial import gen_secret
from Initial import get_ip
from mask import secret_seed
from Hash import SHA256

sys.stdout = Logger('log_user.txt')

p = 109
g = 6
public_a = 77
public_b = 47
private_u = 13

user_number, total_round = para()
r, c = dimension()
buff_size = 6553500

# Each client creates a random seed corresponding to the server
seed_0 = gen_secret(user_number, private_u, public_a)
seed_1 = gen_secret(user_number, private_u, public_b)
Addr = get_ip()


def main():
    global seed_0, seed_1

    user0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    user0.connect((Addr, port_user_alpha()))

    user1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    user1.connect((Addr, port_user_beta()))

    user0.send(str(c).encode('utf-8'))
    user1.send(str(c).encode('utf-8'))

    time_total_pm = 0

    for e in range(total_round):
        print('*****************')
        print('Start round %d' % int(e + 1))
        print('*****************')

        time_pm = 0

        # Local Train
        data = train(user_number, r, c)

        for j in range(user_number):
            sendData = json.dumps(data[j], cls=NpEncoder)
            if j == 0:
                print("users starts training")
                print('The total size of the local gradient：%d KB' % (len(sendData) / 1024))

            time0 = time.perf_counter()
            # Partition and Masking
            data_split_0, data_split_1 = np.array_split(data[j], 2, axis=1)
            data_split_0 = data_split_0 + secret_seed(seed_1[j], data_split_0.shape[0], data_split_0.shape[1])
            data_split_1 = data_split_1 + secret_seed(seed_0[j], data_split_1.shape[0], data_split_1.shape[1])

            sha256 = SHA256()  # Update secrets
            seed_0[j] = (int(sha256.hash(str(seed_0[j])), 16) // 10 ** 72)
            seed_1[j] = (int(sha256.hash(str(seed_1[j])), 16) // 10 ** 72)
            time1 = time.perf_counter()
            time_pm += (time1 - time0)

            sendData_0 = json.dumps(data_split_0, cls=NpEncoder)
            sendData_1 = json.dumps(data_split_1, cls=NpEncoder)
            user0.send(sendData_0.encode('utf-8'))
            user1.send(sendData_1.encode('utf-8'))

            tLen_0 = (len(sendData_0) / 1024)
            tLen_1 = (len(sendData_1) / 1024)

            if j == 0:
                print('The local gradient sent to the S_alpha--%dKB--' % tLen_0)
                print('The local gradient sent to the S_beta--%dKB--' % tLen_1)
                print("-----")

        print('the cryptographic overhead of users in round %d：%f ms' % (e + 1, (time_pm * 1000) / user_number))
        print('-----')
        time_total_pm += (time_pm * 1000) / user_number

        recvData = np.array(json.loads(user1.recv(int(buff_size))))

    print('+++++++++++')
    print('Users, The average time of the cryptographic overhead in all rounds：%f ms' % (time_total_pm / total_round))
    print('+++++++++++')
    user0.close()
    user1.close()


if __name__ == '__main__':
    main()
