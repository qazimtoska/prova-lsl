import torch
from torch.utils.data import Dataset

class EEGSegment(Dataset):
    def __init__(self, data_tensor):
        self.X = data_tensor
        self.y = torch.tensor([0], dtype=torch.long)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]