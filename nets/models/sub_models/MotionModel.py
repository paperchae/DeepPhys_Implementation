import torch
from torch.nn import Module
from torch.nn.functional import normalize


class MotionModel_2D(Module):
    # in_channel = 3, out_channel = 32, kernel_size = (3,3)
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        self.m_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size)
        self.m_batch_Norm1 = torch.nn.BatchNorm2d(out_channels)
        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        self.m_conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                       kernel_size=kernel_size)
        self.m_batch_Norm2 = torch.nn.BatchNorm2d(out_channels)
        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        self.m_avg_pool1 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Dropout1 0.5
        self.m_drop1 = torch.nn.Dropout2d(p=0.5)
        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        self.m_conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2,
                                       kernel_size=kernel_size)
        self.m_batch_Norm3 = torch.nn.BatchNorm2d(out_channels * 2)
        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        self.m_conv4 = torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                       kernel_size=kernel_size)
        self.m_batch_Norm4 = torch.nn.BatchNorm2d(out_channels * 2)
        # Average-Pooling 2x2 kernel layer 6 (64@18x18 -> 64@9x9)
        self.m_avg_pool2 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Dropout2 0.5
        self.m_drop2 = torch.nn.Dropout2d(p=0.5)

    def forward(self, inputs, attention1, attention2):
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        m1 = torch.tanh(self.m_batch_Norm1(self.m_conv1(inputs)))
        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        m2 = torch.tanh(self.m_batch_Norm2(self.m_conv2(m1)))
        # Elementwise multiplication with attention mask 1
        l1norm1 = torch.nn.functional.normalize(attention1)
        e1 = torch.tanh(torch.mul(m2, l1norm1))
        # Dropout 1 0.5
        m3 = self.m_drop1(e1)
        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        m4 = self.m_avg_pool1(m3)

        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        m5 = torch.tanh(self.m_batchnorm3(self.m_conv1(m4)))
        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        m6 = torch.tanh(self.m_batchnorm4(self.m_conv1(m5)))
        # Elementwise multiplication with attention mask 1
        l1norm2 = torch.nn.functional.normalize(attention2)
        e2 = torch.tanh(torch.mul(m6, l1norm2))
        # Dropout 2 0.5
        m7 = self.m_drop1(e2)
        # Average-Pooling 2x2 kernel layer 6 (64@18x18 -> 64@9x9)
        m8 = self.m_avg_pool2(m7)

        return m8
