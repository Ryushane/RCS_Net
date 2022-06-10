import ipdb
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

datapath = r'./handwriting'
labelpath = r'./train.txt'

class MyDataset(Dataset):
    def __init__(self,txtpath):
        imgs = []
        datainfo = open(txtpath,'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0],words[1]))

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        pic, label = self.imgs[index]
        pic = Image.open(datapath+'/'+pic)
        pic = transforms.ToTensor()(pic)
        return pic,label

data = MyDataset(labelpath)

data_loader = DataLoader(data,batch_size=2,shuffle=False,num_workers=0)

ipdb.set_trace()
for pics,label in data_loader:
    print(pics,label)
