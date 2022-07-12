import torch
from torch.nn import Module
import torch.nn.functional as F


class MotionModel_2D(Module):
    # in_channel = 3, out_channel = 32, kernel_size = (3,3)
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        self.m_conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Tanh())

        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        self.m_conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Tanh())

        # Dropout 1
        self.m_dropout1 = torch.nn.Dropout2d(p=0.5)
        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        self.m_avg_pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        self.m_conv_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels * 2),
            torch.nn.Tanh())

        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        self.m_conv_layer4 = torch.nn.Seqential(
            torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels * 2),
            torch.nn.Tanh())

        # Dropout 2
        self.m_dropout2 = torch.nn.Dropout2d(p=0.5)
        # Average-Pooling 2x2 kernel layer 6 (64@18x18 -> 64@9x9)
        self.m_avg_pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, inputs, attention1, attention2):
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        m1 = self.m_conv_layer1(inputs)
        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        m2 = self.m_conv_layer2(m1)
        # Elementwise multiplication with attention mask 1
        l1norm1 = F.normalize(attention1)
        masked1 = torch.tanh(torch.mul(m2, l1norm1))
        # Dropout 1 0.5
        m3 = self.m_drop1(masked1)
        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        m4 = self.m_avg_pool1(m3)

        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        m5 = self.m_conv_layer3(m4)
        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        m6 = self.m_conv_layer4(m5)
        # Elementwise multiplication with attention mask 1
        l1norm2 = F.normalize(attention2)
        masked2 = torch.tanh(torch.mul(m6, l1norm2))
        # Dropout 2 0.5
        m7 = self.m_dropout2(masked2)
        # Average-Pooling 2x2 kernel layer 6 (64@18x18 -> 64@9x9)
        m8 = self.m_avg_pool2(m7)

        return m8
