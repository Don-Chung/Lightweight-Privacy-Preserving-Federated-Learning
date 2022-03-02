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
buff_size = 1000000000
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

    features_0_weight, features_0_bias, features_1_weight, features_1_bias, features_3_weight, features_3_bias, \
    features_4_weight, features_4_bias, features_7_weight, features_7_bias, features_8_weight, features_8_bias, \
    features_10_weight, features_10_bias, features_11_weight, features_11_bias, features_14_weight, features_14_bias, \
    features_15_weight, features_15_bias, features_17_weight, features_17_bias, features_18_weight, features_18_bias, \
    features_20_weight, features_20_bias, features_21_weight, features_21_bias, features_24_weight, features_24_bias, \
    features_25_weight, features_25_bias, features_27_weight, features_27_bias, features_28_weight, features_28_bias, \
    features_30_weight, features_30_bias, features_31_weight, features_31_bias, features_34_weight, features_34_bias, \
    features_35_weight, features_35_bias, features_37_weight, features_37_bias, features_38_weight, features_38_bias, \
    features_40_weight, features_40_bias, features_41_weight, features_41_bias, classifier_weight, classifier_bias = init_cnn_alpha()

    time_avg = 0
    for i in range(user_number):
        decode = {}
        for j in range(54):
            data = tcpCliSock.recv(int(buff_size))
            data = np.array(json.loads(data))

            decode.setdefault(j, data)
        features_0_weight += decode[0] * (size[i] / total_size)
        features_0_bias += decode[1] * (size[i] / total_size)
        features_1_weight += decode[2] * (size[i] / total_size)
        features_1_bias += decode[3] * (size[i] / total_size)
        features_3_weight += decode[4] * (size[i] / total_size)
        features_3_bias += decode[5] * (size[i] / total_size)
        features_4_weight += decode[6] * (size[i] / total_size)
        features_4_bias += decode[7] * (size[i] / total_size)

        features_7_weight += decode[8] * (size[i] / total_size)
        features_7_bias += decode[9] * (size[i] / total_size)
        features_8_weight += decode[10] * (size[i] / total_size)
        features_8_bias += decode[11] * (size[i] / total_size)
        features_10_weight += decode[12] * (size[i] / total_size)
        features_10_bias += decode[13] * (size[i] / total_size)
        features_11_weight += decode[14] * (size[i] / total_size)
        features_11_bias += decode[15] * (size[i] / total_size)

        features_14_weight += decode[16] * (size[i] / total_size)
        features_14_bias += decode[17] * (size[i] / total_size)
        features_15_weight += decode[18] * (size[i] / total_size)
        features_15_bias += decode[19] * (size[i] / total_size)
        features_17_weight += decode[20] * (size[i] / total_size)
        features_17_bias += decode[21] * (size[i] / total_size)
        features_18_weight += decode[22] * (size[i] / total_size)
        features_18_bias += decode[23] * (size[i] / total_size)

        features_20_weight += decode[24] * (size[i] / total_size)
        features_20_bias += decode[25] * (size[i] / total_size)
        features_21_weight += decode[26] * (size[i] / total_size)
        features_21_bias += decode[27] * (size[i] / total_size)
        features_24_weight += decode[28] * (size[i] / total_size)
        features_24_bias += decode[29] * (size[i] / total_size)
        features_25_weight += decode[30] * (size[i] / total_size)
        features_25_bias += decode[31] * (size[i] / total_size)

        features_27_weight += decode[32] * (size[i] / total_size)
        features_27_bias += decode[33] * (size[i] / total_size)
        features_28_weight += decode[34] * (size[i] / total_size)
        features_28_bias += decode[35] * (size[i] / total_size)
        features_30_weight += decode[36] * (size[i] / total_size)
        features_30_bias += decode[37] * (size[i] / total_size)
        features_31_weight += decode[38] * (size[i] / total_size)
        features_31_bias += decode[39] * (size[i] / total_size)

        features_34_weight += decode[40] * (size[i] / total_size)
        features_34_bias += decode[41] * (size[i] / total_size)
        features_35_weight += decode[42] * (size[i] / total_size)
        features_35_bias += decode[43] * (size[i] / total_size)
        features_37_weight += decode[44] * (size[i] / total_size)
        features_37_bias += decode[45] * (size[i] / total_size)
        features_38_weight += decode[46] * (size[i] / total_size)
        features_38_bias += decode[47] * (size[i] / total_size)

        features_40_weight += decode[48] * (size[i] / total_size)
        features_40_bias += decode[49] * (size[i] / total_size)
        features_41_weight += decode[50] * (size[i] / total_size)
        features_41_bias += decode[51] * (size[i] / total_size)
        classifier_weight += decode[52] * (size[i] / total_size)
        classifier_bias += decode[53] * (size[i] / total_size)

    sendData = {0: features_0_weight, 1: features_0_bias, 2: features_1_weight, 3: features_1_bias, 4: features_3_weight, 5: features_3_bias,
                6: features_4_weight, 7: features_4_bias, 8: features_7_weight, 9: features_7_bias, 10: features_8_weight, 11: features_8_bias,
                12: features_10_weight, 13: features_10_bias, 14: features_11_weight, 15: features_11_bias, 16: features_14_weight, 17: features_14_bias,
                18: features_15_weight, 19: features_15_bias, 20: features_17_weight, 21: features_17_bias, 22: features_18_weight, 23: features_18_bias,
                24: features_20_weight, 25: features_20_bias, 26: features_21_weight, 27: features_21_bias, 28: features_24_weight, 29: features_24_bias,
                30: features_25_weight, 31: features_25_bias, 32: features_27_weight, 33: features_27_bias, 34: features_28_weight, 35: features_28_bias,
                36: features_30_weight, 37: features_30_bias, 38: features_31_weight, 39: features_31_bias, 40: features_34_weight, 41: features_34_bias,
                42: features_35_weight, 43: features_35_bias, 44: features_37_weight, 45: features_37_bias, 46: features_38_weight, 47: features_38_bias,
                48: features_40_weight, 49: features_40_bias, 50: features_41_weight, 51: features_41_bias, 52: classifier_weight, 53: classifier_bias}

    tLen = 0
    for i in range(54):
        send_result = json.dumps(sendData[i], cls=NpEncoder)
        tcpCliSock.send(send_result.encode('utf-8'))
        tLen += (len(send_result) / 1024)
        time.sleep(2)
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
