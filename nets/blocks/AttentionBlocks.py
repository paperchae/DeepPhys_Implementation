import torch
from torch.nn import Module
from torch import linalg as LA


class AttentionBlock(Module):
    def __init__(self, in_channels):
        # 1x1 Convolution filter
        self.attention = torch.nn.Conv2d(in_channels=in_channels, out_channels=1,
                                         kernel_size=1, stride=1, padding=1)

    def forward(self, inputs):
        at = torch.sigmoid(self.attention(inputs))
        l1norm = LA.norm(at, 1)
        batchN, channelOut, heightOut, widthOut = inputs.shape
        at = torch.div(heightOut * widthOut * at, 2 * l1norm)

        return at
