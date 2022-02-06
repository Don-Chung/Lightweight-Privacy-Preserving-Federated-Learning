import numpy as np

from Hash import SHA256

train_seed = 50
data = {}
sha256 = SHA256()


def train(user_num, r, c):
    global train_seed
    for i in range(user_num):
        np.random.seed(train_seed)
        data.setdefault(i, np.random.rand(r, c))
        train_seed = (int(sha256.hash(str(train_seed)), 16) // 10 ** 72)
    return data
