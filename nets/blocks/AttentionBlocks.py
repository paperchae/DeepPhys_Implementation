import torch
from torch.nn import Module


class AttentionBlock(Module):
    def __init__(self, in_channels):
        self.attention = torch.nn.Conv2d(in_channels=in_channels, out_channels=1,
                                         kernel_size=(1, 1), stride=(1, 1))

    def forward(self, input):
        at1 = self.attention(input)
        at2 = torch.sigmoid(at1)
        l1norm = torch.norm(at2, p=1)
        batchN, channelOut, heightOut, widthOut = input.shape
        at3 = torch.div(heightOut * widthOut * at2, 2 * l1norm)

        return at3
