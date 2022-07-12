import torch
from torch.utils.data import Dataset
import numpy as np

# class CustomDataset(Dataset):
#     def __init__(self):
#         self.x_data = torch.FloatTensor(x_data)
#         self.y_data = torch.FloatTensor(y_data)
#
#     def __len__(self):
#         return len(self.y_data)
#
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
#
#
# dataset = CustomDataset()

class DeepPhysDataset(Dataset):
    def __init__(self, appearance_data, motion_data, label):
        self.a_data = appearance_data
        self.m_data = motion_data
        self.label = label

    def __getitem__(self, index):
        appearance_data = torch.Tensor(self.a_data[index], dtype=torch.float32)
        motion_data = torch.Tensor(self.m_data[index], dtype=torch.float32)
        label = torch.Tensor(self.label[index], dtype=torch.float32)

        inputs = torch.cat()
        return inputs, label
