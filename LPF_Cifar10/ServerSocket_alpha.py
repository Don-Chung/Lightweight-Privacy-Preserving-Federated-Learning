import random
import socket
import sys
import time

import numpy as np
import json

from Hash import SHA256
from Initial import gen_secret, para, port_user_alpha, port_alpha_beta, init_cnn_alpha, para_dynamic, Logger, \
    gen_secret_join, dynamic_number, dynamic_iod, init_cnn_alpha_r
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
buff_size = 1000000000
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
    features_0_weight, features_0_bias, features_1_weight, features_1_bias, features_3_weight, features_3_bias, \
    features_4_weight, features_4_bias, features_7_weight, features_7_bias, features_8_weight, features_8_bias, \
    features_10_weight, features_10_bias, features_11_weight, features_11_bias, features_14_weight, features_14_bias, \
    features_15_weight, features_15_bias, features_17_weight, features_17_bias, features_18_weight, features_18_bias, \
    features_20_weight, features_20_bias, features_21_weight, features_21_bias, features_24_weight, features_24_bias, \
    features_25_weight, features_25_bias, features_27_weight, features_27_bias, features_28_weight, features_28_bias, \
    features_30_weight, features_30_bias, features_31_weight, features_31_bias, features_34_weight, features_34_bias, \
    features_35_weight, features_35_bias, features_37_weight, features_37_bias, features_38_weight, features_38_bias, \
    features_40_weight, features_40_bias, features_41_weight, features_41_bias, classifier_weight, classifier_bias = init_cnn_alpha()

    features_r0_weight, features_r0_bias, features_r1_weight, features_r1_bias, features_r3_weight, features_r3_bias, \
    features_r4_weight, features_r4_bias, features_r7_weight, features_r7_bias, features_r8_weight, features_r8_bias, \
    features_r10_weight, features_r10_bias, features_r11_weight, features_r11_bias, features_r14_weight, features_r14_bias, \
    features_r15_weight, features_r15_bias, features_r17_weight, features_r17_bias, features_r18_weight, features_r18_bias, \
    features_r20_weight, features_r20_bias, features_r21_weight, features_r21_bias, features_r24_weight, features_r24_bias, \
    features_r25_weight, features_r25_bias, features_r27_weight, features_r27_bias, features_r28_weight, features_r28_bias, \
    features_r30_weight, features_r30_bias, features_r31_weight, features_r31_bias, features_r34_weight, features_r34_bias, \
    features_r35_weight, features_r35_bias, features_r37_weight, features_r37_bias, features_r38_weight, features_r38_bias, \
    features_r40_weight, features_r40_bias, features_r41_weight, features_r41_bias, classifier_rweight, classifier_rbias = init_cnn_alpha_r()

    for i in range(user_number):
        decode = {}
        for j in range(54):
            data = tcpCliSock.recv(int(buff_size))
            data = np.array(json.loads(data))

            decode.setdefault(j, data)
        # Federal average
        time0 = time.perf_counter()

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

        features_r0_weight += (- secret_seed(seed_0[i], decode[0].shape[0], decode[0].shape[1]+1)) * (size[i] / total_size)
        features_r0_bias += (- secret_seed(seed_0[i], decode[1].shape[0], 0)) * (size[i] / total_size)
        features_r1_weight += (- secret_seed(seed_0[i], decode[2].shape[0], 0)) * (size[i] / total_size)
        features_r1_bias += (- secret_seed(seed_0[i], decode[3].shape[0], 0)) * (size[i] / total_size)
        features_r3_weight += (- secret_seed(seed_0[i], decode[4].shape[0], decode[4].shape[1]+1)) * (size[i] / total_size)
        features_r3_bias += (- secret_seed(seed_0[i], decode[5].shape[0], 0)) * (size[i] / total_size)
        features_r4_weight += (- secret_seed(seed_0[i], decode[6].shape[0], 0)) * (size[i] / total_size)
        features_r4_bias += (- secret_seed(seed_0[i], decode[7].shape[0], 0)) * (size[i] / total_size)

        features_r7_weight += (- secret_seed(seed_0[i], decode[8].shape[0], decode[8].shape[1]+1)) * (size[i] / total_size)
        features_r7_bias += (- secret_seed(seed_0[i], decode[9].shape[0], 0)) * (size[i] / total_size)
        features_r8_weight += (- secret_seed(seed_0[i], decode[10].shape[0], 0)) * (size[i] / total_size)
        features_r8_bias += (- secret_seed(seed_0[i], decode[11].shape[0], 0)) * (size[i] / total_size)
        features_r10_weight += (- secret_seed(seed_0[i], decode[12].shape[0], decode[12].shape[1]+1)) * (size[i] / total_size)
        features_r10_bias += (- secret_seed(seed_0[i], decode[13].shape[0], 0)) * (size[i] / total_size)
        features_r11_weight += (- secret_seed(seed_0[i], decode[14].shape[0], 0)) * (size[i] / total_size)
        features_r11_bias += (- secret_seed(seed_0[i], decode[15].shape[0], 0)) * (size[i] / total_size)

        features_r14_weight += (- secret_seed(seed_0[i], decode[16].shape[0], decode[16].shape[1]+1)) * (size[i] / total_size)
        features_r14_bias += (- secret_seed(seed_0[i], decode[17].shape[0], 0)) * (size[i] / total_size)
        features_r15_weight += (- secret_seed(seed_0[i], decode[18].shape[0], 0)) * (size[i] / total_size)
        features_r15_bias += (- secret_seed(seed_0[i], decode[19].shape[0], 0)) * (size[i] / total_size)
        features_r17_weight += (- secret_seed(seed_0[i], decode[20].shape[0], decode[20].shape[1]+1)) * (size[i] / total_size)
        features_r17_bias += (- secret_seed(seed_0[i], decode[21].shape[0], 0)) * (size[i] / total_size)
        features_r18_weight += (- secret_seed(seed_0[i], decode[22].shape[0], 0)) * (size[i] / total_size)
        features_r18_bias += (- secret_seed(seed_0[i], decode[23].shape[0], 0)) * (size[i] / total_size)

        features_r20_weight += (- secret_seed(seed_0[i], decode[24].shape[0], decode[24].shape[1]+1)) * (size[i] / total_size)
        features_r20_bias += (- secret_seed(seed_0[i], decode[25].shape[0], 0)) * (size[i] / total_size)
        features_r21_weight += (- secret_seed(seed_0[i], decode[26].shape[0], 0)) * (size[i] / total_size)
        features_r21_bias += (- secret_seed(seed_0[i], decode[27].shape[0], 0)) * (size[i] / total_size)
        features_r24_weight += (- secret_seed(seed_0[i], decode[28].shape[0], decode[28].shape[1]+1)) * (size[i] / total_size)
        features_r24_bias += (- secret_seed(seed_0[i], decode[29].shape[0], 0)) * (size[i] / total_size)
        features_r25_weight += (- secret_seed(seed_0[i], decode[30].shape[0], 0)) * (size[i] / total_size)
        features_r25_bias += (- secret_seed(seed_0[i], decode[31].shape[0], 0)) * (size[i] / total_size)

        features_r27_weight += (- secret_seed(seed_0[i], decode[32].shape[0], decode[32].shape[1]+1)) * (size[i] / total_size)
        features_r27_bias += (- secret_seed(seed_0[i], decode[33].shape[0], 0)) * (size[i] / total_size)
        features_r28_weight += (- secret_seed(seed_0[i], decode[34].shape[0], 0)) * (size[i] / total_size)
        features_r28_bias += (- secret_seed(seed_0[i], decode[35].shape[0], 0)) * (size[i] / total_size)
        features_r30_weight += (- secret_seed(seed_0[i], decode[36].shape[0], decode[36].shape[1]+1)) * (size[i] / total_size)
        features_r30_bias += (- secret_seed(seed_0[i], decode[37].shape[0], 0)) * (size[i] / total_size)
        features_r31_weight += (- secret_seed(seed_0[i], decode[38].shape[0], 0)) * (size[i] / total_size)
        features_r31_bias += (- secret_seed(seed_0[i], decode[39].shape[0], 0)) * (size[i] / total_size)

        features_r34_weight += (- secret_seed(seed_0[i], decode[40].shape[0], decode[40].shape[1]+1)) * (size[i] / total_size)
        features_r34_bias += (- secret_seed(seed_0[i], decode[41].shape[0], 0)) * (size[i] / total_size)
        features_r35_weight += (- secret_seed(seed_0[i], decode[42].shape[0], 0)) * (size[i] / total_size)
        features_r35_bias += (- secret_seed(seed_0[i], decode[43].shape[0], 0)) * (size[i] / total_size)
        features_r37_weight += (- secret_seed(seed_0[i], decode[44].shape[0], decode[44].shape[1]+1)) * (size[i] / total_size)
        features_r37_bias += (- secret_seed(seed_0[i], decode[45].shape[0], 0)) * (size[i] / total_size)
        features_r38_weight += (- secret_seed(seed_0[i], decode[46].shape[0], 0)) * (size[i] / total_size)
        features_r38_bias += (- secret_seed(seed_0[i], decode[47].shape[0], 0)) * (size[i] / total_size)

        features_r40_weight += (- secret_seed(seed_0[i], decode[48].shape[0], decode[48].shape[1]+1)) * (size[i] / total_size)
        features_r40_bias += (- secret_seed(seed_0[i], decode[49].shape[0], 0)) * (size[i] / total_size)
        features_r41_weight += (- secret_seed(seed_0[i], decode[50].shape[0], 0)) * (size[i] / total_size)
        features_r41_bias += (- secret_seed(seed_0[i], decode[51].shape[0], 0)) * (size[i] / total_size)
        classifier_rweight += (- secret_seed(seed_0[i], decode[52].shape[0], decode[52].shape[1])) * (size[i] / total_size)
        classifier_rbias += (- secret_seed(seed_0[i], decode[53].shape[0], 0)) * (size[i] / total_size)

        time1 = time.perf_counter()
        time_avg += (time1 - time0)

    timef0 = time.perf_counter()

    features_0_weight = np.hstack((features_0_weight, features_r0_weight))
    features_0_bias = np.hstack((features_0_bias, features_r0_bias))
    features_1_weight = np.hstack((features_1_weight, features_r1_weight))
    features_1_bias = np.hstack((features_1_bias, features_r1_bias))
    features_3_weight = np.hstack((features_3_weight, features_r3_weight))
    features_3_bias = np.hstack((features_3_bias, features_r3_bias))
    features_4_weight = np.hstack((features_4_weight, features_r4_weight))
    features_4_bias = np.hstack((features_4_bias, features_r4_bias))

    features_7_weight = np.hstack((features_7_weight, features_r7_weight))
    features_7_bias = np.hstack((features_7_bias, features_r7_bias))
    features_8_weight = np.hstack((features_8_weight, features_r8_weight))
    features_8_bias = np.hstack((features_8_bias, features_r8_bias))
    features_10_weight = np.hstack((features_10_weight, features_r10_weight))
    features_10_bias = np.hstack((features_10_bias, features_r10_bias))
    features_11_weight = np.hstack((features_11_weight, features_r11_weight))
    features_11_bias = np.hstack((features_11_bias, features_r11_bias))

    features_14_weight = np.hstack((features_14_weight, features_r14_weight))
    features_14_bias = np.hstack((features_14_bias, features_r14_bias))
    features_15_weight = np.hstack((features_15_weight, features_r15_weight))
    features_15_bias = np.hstack((features_15_bias, features_r15_bias))
    features_17_weight = np.hstack((features_17_weight, features_r17_weight))
    features_17_bias = np.hstack((features_17_bias, features_r17_bias))
    features_18_weight = np.hstack((features_18_weight, features_r18_weight))
    features_18_bias = np.hstack((features_18_bias, features_r18_bias))

    features_20_weight = np.hstack((features_20_weight, features_r20_weight))
    features_20_bias = np.hstack((features_20_bias, features_r20_bias))
    features_21_weight = np.hstack((features_21_weight, features_r21_weight))
    features_21_bias = np.hstack((features_21_bias, features_r21_bias))
    features_24_weight = np.hstack((features_24_weight, features_r24_weight))
    features_24_bias = np.hstack((features_24_bias, features_r24_bias))
    features_25_weight = np.hstack((features_25_weight, features_r25_weight))
    features_25_bias = np.hstack((features_25_bias, features_r25_bias))

    features_27_weight = np.hstack((features_27_weight, features_r27_weight))
    features_27_bias = np.hstack((features_27_bias, features_r27_bias))
    features_28_weight = np.hstack((features_28_weight, features_r28_weight))
    features_28_bias = np.hstack((features_28_bias, features_r28_bias))
    features_30_weight = np.hstack((features_30_weight, features_r30_weight))
    features_30_bias = np.hstack((features_30_bias, features_r30_bias))
    features_31_weight = np.hstack((features_31_weight, features_r31_weight))
    features_31_bias = np.hstack((features_31_bias, features_r31_bias))

    features_34_weight = np.hstack((features_34_weight, features_r34_weight))
    features_34_bias = np.hstack((features_34_bias, features_r34_bias))
    features_35_weight = np.hstack((features_35_weight, features_r35_weight))
    features_35_bias = np.hstack((features_35_bias, features_r35_bias))
    features_37_weight = np.hstack((features_37_weight, features_r37_weight))
    features_37_bias = np.hstack((features_37_bias, features_r37_bias))
    features_38_weight = np.hstack((features_38_weight, features_r38_weight))
    features_38_bias = np.hstack((features_38_bias, features_r38_bias))

    features_40_weight = np.hstack((features_40_weight, features_r40_weight))
    features_40_bias = np.hstack((features_40_bias, features_r40_bias))
    features_41_weight = np.hstack((features_41_weight, features_r41_weight))
    features_41_bias = np.hstack((features_41_bias, features_r41_bias))
    classifier_weight = np.hstack((classifier_weight, classifier_rweight))
    classifier_bias = np.hstack((classifier_bias, classifier_rbias))

    timef1 = time.perf_counter()

    sendData = {0: features_0_weight, 1: features_0_bias, 2: features_1_weight, 3: features_1_bias,
                4: features_3_weight, 5: features_3_bias,
                6: features_4_weight, 7: features_4_bias, 8: features_7_weight, 9: features_7_bias,
                10: features_8_weight, 11: features_8_bias,
                12: features_10_weight, 13: features_10_bias, 14: features_11_weight, 15: features_11_bias,
                16: features_14_weight, 17: features_14_bias,
                18: features_15_weight, 19: features_15_bias, 20: features_17_weight, 21: features_17_bias,
                22: features_18_weight, 23: features_18_bias,
                24: features_20_weight, 25: features_20_bias, 26: features_21_weight, 27: features_21_bias,
                28: features_24_weight, 29: features_24_bias,
                30: features_25_weight, 31: features_25_bias, 32: features_27_weight, 33: features_27_bias,
                34: features_28_weight, 35: features_28_bias,
                36: features_30_weight, 37: features_30_bias, 38: features_31_weight, 39: features_31_bias,
                40: features_34_weight, 41: features_34_bias,
                42: features_35_weight, 43: features_35_bias, 44: features_37_weight, 45: features_37_bias,
                46: features_38_weight, 47: features_38_bias,
                48: features_40_weight, 49: features_40_bias, 50: features_41_weight, 51: features_41_bias,
                52: classifier_weight, 53: classifier_bias}
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
    for i in range(54):
        send_result = json.dumps(sendData[i], cls=NpEncoder)
        StS.send(send_result.encode('utf-8'))
        tLen += (len(send_result) / 1024)
        time.sleep(2)
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
