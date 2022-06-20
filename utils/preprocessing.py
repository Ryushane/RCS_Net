from tkinter.messagebox import NO
import ipdb
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np


# labelpath = r'./train.txt'


class RCSdataset(Dataset):
    def __init__(self, txt, transform=None, target_trasfrom=None):
        with open(txt, 'r') as datainfo:
            datasets = []
            for line in datainfo:
                line = line.strip('\n')
                words = line.split()
                datasets.append((words[0], int(words[1])))

        self.datasets = datasets
        self.transform = transform
        self.target_trasfrom = target_trasfrom


    def __getitem__(self,index):
        fn, label = self.datasets[index]
        sig = loadmat(fn)
        # ipdb.set_trace()
        if self.transform is not None:
            sig = sig['temp_data'].astype(np.float32)
            sig = sig[:,0] # Only reserve the value of RCS
            sig = sig[np.newaxis, :] # Add a new dimension for RCS
            # max_data = max(sig)
            # db3 = max_data/(2**0.5)
            # max_index = []
            # for i in range(len(sig)):
            #     if sig[i]>db3:
            #         max_index.append(i)
            # middle = sum(max_index)/len(max_index)
            ####insert####
            # left = int((int(middle)-64)//4)*4
            # right = int((int(middle)+64)//4)*4
            # sig = sig[left:right]
            # sig = np.append(sig[::4],[middle])
            ####origin####
            #left = int(middle)-16
            #right = int(middle)+16
            #sig = sig[left:right]
            #sig = np.append(sig,[middle])
            # sig = self.transform(sig)
        return sig, label

    def __len__(self):
        return len(self.datasets)

if __name__ == '__main__':
    root = './'   # root

    train_data = RCSdataset(txt=root+'/'+'train.txt', transform=transforms.ToTensor())
    test_data = RCSdataset(txt=root+'/'+'test.txt', transform=transforms.ToTensor())


    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64)

    for data, label in train_loader:
        print(data, label)
        # ipdb.set_trace()

