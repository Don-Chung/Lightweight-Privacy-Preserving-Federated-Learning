from __future__ import print_function  # 从future版本导入print函数功能
import argparse  # 加载处理命令行参数的库
import copy
import pickle

import torch  # 引入相关的包
import torch.nn.functional as F  # 引用神经网络常用函数包，不具有可学习的参数
import torch.optim as optim
from torchvision import datasets, transforms  # 加载pytorch官方提供的dataset
import torch.utils.data as Data
from torch.utils.data import DataLoader


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)  # negative log likelihood loss(nll_loss), sum up batch cross entropy
            loss.backward()
            optimizer.step()  # 根据parameter的梯度更新parameter


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 无需计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return 100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def LocalTrain(GlobalModel, num):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    f = open('./data/users/' + str(num + 1) + '_user.pkl', 'rb')
    images_i, labels_i = pickle.load(f)
    f.close()
    images_i, labels_i = images_i.to(device), labels_i.to(device)
    data = Data.TensorDataset(images_i, labels_i)
    train_loader = DataLoader(data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = copy.deepcopy(GlobalModel).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)  # optimizer存储了所有parameters的引用，每个parameter都包含gradient
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24], gamma=0.1)  # 学习率按区间更新

    # for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, 10)
    accuracy = test(args, model, device, test_loader)

    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_fc.pt")
    numpy_para = {}
    i = 0
    parameters = model.parameters()
    for p in parameters:
        numpy_para.setdefault(i, p.detach().cpu().numpy())
        i = i + 1
    return numpy_para, accuracy
