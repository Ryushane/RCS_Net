import torch.nn as nn
import torch
import ipdb
import numpy as np


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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # ipdb.set_trace()
        x = self.features(x)
        x = x.view(x.size(0), 638848)
        out = self.classifier(x)

        return out