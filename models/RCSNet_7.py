import torch.nn as nn
import torch
import ipdb
import numpy as np


class WALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WALinear, self).__init__()
        self.in_features = in_features           # input features
        self.out_features = out_features         # class num
        self.sub_num_classes = 3     # only 3 class in this situation
        self.WA_linears = nn.ModuleList()
        self.WA_linears.extend([nn.Linear(self.in_features, self.sub_num_classes, bias=False) for i in range(3)])

    def forward(self, x):
        out1 = self.WA_linears[0](x)
        out2 = self.WA_linears[1](x)
        out3 = self.WA_linears[2](x)

        return torch.cat([out1, out2, out3], dim = 1)
    
    def align_norms(self, step_b):    # step_b代表着增量步骤
        # Fetch old and new layers
        new_layer = self.WA_linears[step_b]
        old_layers = self.WA_linears[:step_b]
        
        # Get weight of layers
        new_weight = new_layer.weight.cpu().detach().numpy()
        for i in range(step_b):
            old_weight = np.concatenate([old_layers[i].weight.cpu().detach().numpy() for i in range(step_b)])
        print("old_weight's shape is: ",old_weight.shape)
        print("new_weight's shape is: ",new_weight.shape)

        # Calculate the norm
        Norm_of_new = np.linalg.norm(new_weight, axis=1)
        Norm_of_old = np.linalg.norm(old_weight, axis=1)
        assert(len(Norm_of_new) == 1)     # only 1 new class
        assert(len(Norm_of_old) == step_b)     # two old class
        
        # Calculate the Gamma
        gamma = np.mean(Norm_of_new) / np.mean(Norm_of_old)
        print("Gamma = ", gamma)

        # Update new layer's weight
        updated_new_weight = torch.Tensor(gamma * new_weight).cuda()
        print(updated_new_weight)
        self.WA_linears[step_b].weight = torch.nn.Parameter(updated_new_weight)

class RCSNet_7(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # Extract features, 1D conv
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, 7),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 7),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 7),
            nn.ReLU(),
        )
        # Classifiy
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(638848, 128),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(128, num_classes),
            WALinear(128, num_classes)
        )

    def weight_align(self, step_b):
        self.fc.align_norms(step_b)

    def forward(self, x):
        # ipdb.set_trace()
        x = self.features(x)
        x = x.view(x.size(0), 638848)
        out = self.classifier(x)

        return out