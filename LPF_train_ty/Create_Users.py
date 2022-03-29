import pickle
import os
import sys

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

from Initial import para, get_FileSize, Logger

sys.stdout = Logger('log.txt')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

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

    #if num == 1:
    #    size = 150
    #else:
    #    size = random.randint(50, 150)
    multi = 2
    size = (500 // num) * multi

    dataset1 = datasets.FashionMNIST('./data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
    train_loader = DataLoader(dataset1, batch_size=data_of_one_batch * size, shuffle=True)

    #for batch_idx, z in enumerate(train_loader):
    #    if batch_idx > 0:
    #        break

    #filehandler = open(dir + '/' + str(num) + '_user.pkl', "wb")
    #pickle.dump(z, filehandler)
    #filehandler.close()
    #print('client created at: ' + dir + '/' + str(num) + '_user.pkl')
    j = 1
    for i in range(multi):
        for batch_idx, z in enumerate(train_loader):
            if os.path.exists(dir + '/' + str(num) + '_user.pkl'):
                print('Client exists at: ' + dir + '/' + str(num) + '_user.pkl')
                j = j + 1
                continue
            filehandler = open(dir + '/' + str(j) + '_user.pkl', "wb")
            pickle.dump(z, filehandler)
            filehandler.close()
            print('client created at: ' + dir + '/' + str(num) + '_user.pkl')
            # size = get_FileSize('./data/users/' + str(j) + '_user.pkl')
            # print('user -- %d --size: %f MB' % (j, size))
            j = j + 1


if __name__ == '__main__':
    # user_number, total_round = para()
    user_number = 100
    num = user_number
    create_users(num, os.getcwd() + '/data/users')
    #for j in range(1, num + 1):
    #    create_users(j, os.getcwd() + '/data/users')
    #    size = get_FileSize('./data/users/' + str(j) + '_user.pkl')
    #    print('user -- %d --size: %f MB' % (j, size))
