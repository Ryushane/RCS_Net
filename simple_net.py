import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import activation
import torch.nn.functional as F

# 搭建网络
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cl1 = nn.Linear(256, 128)
        self.cl2 = nn.Linear(128,10)
        self.fc1 = nn.Linear(10,3)
    
    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.fc1(x))
        return x

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = MyModel()
model.fc1.register_forward_hook(get_activation('fc1'))
x = torch.randn(1, 256)
output = model(x)
print(activation)
