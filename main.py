import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import argparse
import ipdb

from models import *
from misc import progress_bar
# from config.set_params import params as sp
from utils.preprocessing import RCSdataset
from utils.preprocessing import get_mean_std_value, get_max_min_value

import random
import time
import os
import sys



seed_num = 435
root = "./"
CLASSES = ('car','cycling','walking')

# 固定seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 输出log，方便晚上批量跑实验
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
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
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=32, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=32, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    sys.stdout = Logger(sys.stdout)  #  将输出记录到log
    sys.stderr = Logger(sys.stderr)  # 将错误信息记录到log 
    # 记录运行时间
    torch.cuda.synchronize()
    start = time.time()
    solver = Solver(args)
    solver.run()
    torch.cuda.synchronize()
    end = time.time()
    # ipdb.set_trace()
    print("Running time = " + str(end - start) + '/s')


# RCS归一化方法
class RCSNorm(object):
    """
    将输入的RCS序列数据归一化到[0,1]
    """
    def __init__(self, max_value, min_value):
        self.max_value = max_value
        self.min_value = min_value
    def __call__(self, data):
        range = self.max_value - self.min_value
        data = (data - self.min_value) / range
        return data
    
class Solver(object):
    def __init__(self, config):
        self.model = None
        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):

        input_transform = Compose([
                                ToTensor(),
                                # Normalize([-1.2481],[8.3537])  # 3 Classes
                                Normalize([-2.8032],[8.3217])  # 4 Classes
                                ])

        input_transform_eval = Compose([
                                ToTensor(),
                                # Normalize([-1.2481],[8.3537])  # 3 Class
                                Normalize([-2.8032],[8.3217])  # 4 Classes
                                ])
        # train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        # test_transform = transforms.Compose([transforms.ToTensor()])
        # train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        # self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True) 
        train_set = RCSdataset(txt=root+'/'+'train.txt',transform=input_transform)
        self.train_loader = DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        # mean, std = get_mean_std_value(self.train_loader)
        # max, min = get_max_min_value(self.train_loader)
        # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        # self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)
        test_set = RCSdataset(txt=root+'/'+'test.txt',transform=input_transform_eval)
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
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 100, 150], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        num_classes = self.num_classes

        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            # print(f"Incremental step :  {step_b + 1}")
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
        model_out_path = "./checkpoints/" + "epochs-" + str(self.epochs) + "_lr-" + str(self.lr) +  ".pth"
        # model_out_path = "./checkpoints/model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            print("\n===> epoch: %d/%d" % (epoch,self.epochs))
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()
            self.scheduler.step()

if __name__ == '__main__':
    main()
