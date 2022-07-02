import torch
from torch.nn import Module
from nets.blocks.AttentionBlocks import AttentionBlock

# input :
class AppearanceModel_2D(Module):
    # in_channels = 3 , out_channels = 32, conv_kernel_size = (3,3)
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        self.a_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=1, padding=1)
        self.a_batch_Norm1 = torch.nn.BatchNorm2d(out_channels)

        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        self.a_conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=1, padding=1)
        self.a_batch_Norm2 = torch.nn.BatchNorm2d(out_channels)

        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        self.avg_pool1 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        self.a_conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2,
                                       kernel_size=kernel_size, stride=1, padding=1)
        self.a_batch_Norm3 = torch.nn.BatchNorm2d(out_channels * 2)

        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        self.a_conv4 = torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                       kernel_size=kernel_size, stride=1, padding=1)
        self.a_batch_Norm4 = torch.nn.BatchNorm2d(out_channels * 2)

        # Average-Pooling 2x2 kernel layer 6 (64@18x18 -> 64@9x9)
        self.avg_pool2 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    # Appearance Model forward
    def forward(self, inputs):
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        ap1 = torch.tanh(self.a_batch_Norm1(self.a_conv1(inputs)))
        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        ap2 = torch.tanh(self.a_batch_Norm2(self.a_conv2(ap1)))
        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        ap3 = self.avg_pool1(ap2)
        # Attention 1
        at1 = AttentionBlock(ap2)
        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        ap4 = torch.tanh(self.a_batch_Norm3(self.a_conv3(ap3)))
        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        ap5 = torch.tanh(self.a_batch_Norm4(self.a_conv4(ap4)))
        # Attention 2
        at2 = AttentionBlock(ap5)

        return at1, at2
