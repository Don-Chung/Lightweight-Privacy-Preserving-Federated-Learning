import os
import threading


def server1():
    os.system('python ./ServerSocket_beta.py')


def server0():
    os.system('python ./ServerSocket_alpha.py')


def user():
    os.system('python ./UserSocket.py')


threads = [threading.Thread(target=server1), threading.Thread(target=server0), threading.Thread(target=user)]

if __name__ == '__main__':
    for t in threads:
        t.start()
