import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import argparse
import ipdb

import time
import os
import sys
import random

from models import *
from misc import progress_bar
# from config.set_params import params as sp
from utils.preprocessing import RCSdataset

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker

root = "./"
CLASSES = ('car','cycling','walking')
seed_num = 435

# 固定seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 输出log，方便晚上批量跑实验
class Logger(object):

    def __init__(self, stream=sys.stdout, epochs = None, num_classes = None, lr = None, seed = None):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M') + "_epochs-" + str(epochs) + "_num_classes-" + str(num_classes) + "_lr-" + str(lr) + "_seed-" + str(seed))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    parser = argparse.ArgumentParser(description="RCSNet with PyTorch")
    parser.add_argument('--input_dim', default=1, type=int, help='input dim')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=150, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=32, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=32, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    sys.stdout = Logger(sys.stdout, args.epochs, args.num_classes, args.lr, seed_num)  #  将输出记录到log
    sys.stderr = Logger(sys.stderr, args.epochs, args.num_classes, args.lr, seed_num)  # 将错误信息记录到log 

    solver = Solver(args)
    solver.run()
    solver.plot()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.lr = config.lr
        self.epochs = config.epochs
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        # train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        # self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True) 
        train_set = RCSdataset(txt=root+'/'+'train.txt',transform=transforms.ToTensor())
        self.train_loader = DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        # self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)
        test_set = RCSdataset(txt=root+'/'+'test.txt',transform=transforms.ToTensor())
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.train_batch_size)
        # ipdb.set_trace()

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = LeNet().to(self.device)
        # self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = GoogLeNet().to(self.device)
        # self.model = RCSNet(self.input_dim, self.num_classes).to(self.device)
        self.model = RCSNet_7(self.input_dim, self.num_classes).to(self.device)
        # self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        # self.model = resnet152().to(self.device)
        # self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        # self.model = WideResNet(depth=28, num_classes=10).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
        return test_loss, test_correct / total

    def save(self):
        model_out_path = "./checkpoints/" + "epochs-" + str(self.epochs) + "_lr-" + str(self.lr) + "_seed-" + str(seed_num) + ".pth"
        # model_out_path = "./checkpoints/model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def plot(self):
        config = {
        "font.family":'serif',
        "font.size": 18,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
        }
        rcParams.update(config)

        plt.figure()
        x = np.linspace(1, self.epochs, self.epochs)
        plt.plot(x, self.train_losses, c='b', label='train')
        plt.plot(x, self.test_losses, c='r', label='test')
        plt.xlabel("训练轮次")
        plt.ylabel("loss")
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))   # set x label's axis to integer
        # plt.ylim(0.4, 0.7)
        plt.legend()
        figure_save_path = "file_fig"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
        # plt.savefig(os.path.join(figure_save_path , 'exam.png'))#第一个是指存储路径，第二个是图片名字
        plt.savefig(os.path.join(figure_save_path, "epochs-" + str(self.epochs) + "_lr-" + str(self.lr) + "_seed-" + str(seed_num) +".png"), bbox_inches="tight")
        # plt.show()
        
    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        train_losses =[]
        test_losses =[]
        for epoch in range(1, self.epochs + 1):
            print("\n===> epoch: %d/%d" % (epoch,self.epochs))
            train_result = self.train()
            train_losses.append(train_result[0])
            print(train_result)
            test_result = self.test()
            test_losses.append(test_result[0])
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()
            self.scheduler.step()
        self.train_losses = train_losses
        self.test_losses = test_losses


if __name__ == '__main__':
    main()
