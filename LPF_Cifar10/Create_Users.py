import pickle
import os

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

from Initial import para, get_FileSize, para_dynamic

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_of_one_batch = 100  # 每批数据量


def create_users(num, dir):
    '''
    This function creates clients that hold non-iid MNIST data
    but the way these indices are grouped, they create a non-iid client.)
    '''
    if os.path.exists(dir + '/' + str(num) + '_user.pkl'):
        print('Client exists at: ' + dir + '/' + str(num) + '_user.pkl')
        return
    if not os.path.exists(dir):
        os.makedirs(dir)

    size = random.randint(450, 500)

    dataset1 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform)
    train_loader = DataLoader(dataset1, batch_size=data_of_one_batch * size, shuffle=True)

    for batch_idx, z in enumerate(train_loader):
        if batch_idx == 0:
            break

    filehandler = open(dir + '/' + str(num) + '_user.pkl', "wb")
    pickle.dump(z, filehandler)
    filehandler.close()
    print('client created at: ' + dir + '/' + str(num) + '_user.pkl')


if __name__ == '__main__':
    # user_number, total_round = para()
    # for j in range(1, user_number + 1):
    #     create_users(j, os.getcwd() + '/data/users')
    #     size = get_FileSize('./data/users/' + str(j) + '_user.pkl')
    #     print('user -- %d --size: %f MB' % (j, size))
    user_number, total_round = para()
    dynamic, dynamic_range = para_dynamic()
    if dynamic == 1:
        num = user_number + (user_number * dynamic_range // 100)
    else:
        num = user_number
    for j in range(1, num + 1):
        create_users(j, os.getcwd() + '/data/users')
        size = get_FileSize('./data/users/' + str(j) + '_user.pkl')
        print('user -- %d --size: %f MB' % (j, size))
