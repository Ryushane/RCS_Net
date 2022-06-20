import torch

class params:
    """Modify this class to set parameters."""
    def __init__(self):
        self.params = {
            "train": "./train.txt",
            "test": "./test.txt",
            "resume": None,
            "input_dim": 1,
            "num_classes": 3,
            # "workers": 4,
            # "batch_size": 32,
            # "epochs": 300,
            # "lr": 0.003,
            # "momentum": 0.9,
            # "weight_decay": 1e-4,
            # "split": 0.2,
            # "use_cuda": torch.cuda.is_available(),
        }