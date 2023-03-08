from tkinter.messagebox import NO
import ipdb
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np


# labelpath = r'./train.txt'
# need to devide data into 3 groups

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
        sig = sig['temp_data'].astype(np.float32)
        sig = sig[:,0] # Only reserve the value of RCS
        sig = sig[np.newaxis, :] # Add a new dimension for RCS
        if self.transform is not None:
            sig = self.transform(sig)
            sig = sig.squeeze(0)
            # ipdb.set_trace()
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

def get_max_min_value(loader):
    data_max, data_min = 0,0

    for data,_ in loader:
        if(data_max <= torch.max(data)):
            data_max = torch.max(data)
        if(data_min >= torch.min(data)):
            data_min = torch.min(data)

    return data_max,data_min

def get_mean_std_value(loader):
    data_sum, data_squared_sum, num_batches = 0,0,0

    for data,_ in loader:
        data_sum += torch.mean(data)
        data_squared_sum += torch.mean(data**2)
        # ipdb.set_trace()
        num_batches += 1
    
    mean = data_sum / num_batches
    std = (data_squared_sum / num_batches - mean**2)**0.5
    return mean,std

if __name__ == '__main__':
    root = './'   # root
    # mean = -1.2481274604797363,std = 8.353983879089355
    # max = 28.404359817504883,min = -65.7984390258789
    input_transform = Compose([
                            ToTensor(),
                            # Normalize([-1.2481],[8.3537])
                            ])

    input_transform_eval = Compose([
                            ToTensor(),
                            # Normalize([-1.2481],[8.3537])
                            ])
    train_data = RCSdataset(txt=root+'/'+'train.txt', transform=input_transform)
    test_data = RCSdataset(txt=root+'/'+'test.txt', transform=input_transform_eval)


    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=64)


    for data, label in train_loader:
        print(data, label)
        # ipdb.set_trace()
    mean,std = get_mean_std_value(train_loader)
    data_max, data_min = get_max_min_value(train_loader)
    print('mean = {},std = {}'.format(mean,std))
    print('max = {},min = {}'.format(data_max,data_min))

