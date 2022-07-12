import torch
from torch.nn import Module
from nets.blocks.AttentionBlocks import AttentionBlock

class AppearanceModel_2D(Module):
    # in_channels = 3 , out_channels = 32, conv_kernel_size = (3,3)
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        self.a_conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Tanh())

        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        self.a_conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Tanh())

        # Dropout
        self.a_dropout = torch.nn.Dropout2d(p=0.5)
        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        self.avg_pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        self.a_conv_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels * 2),
            torch.nn.Tanh())

        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        self.a_conv_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                            kernel_size=kernel_size, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels * 2),
            torch.nn.Tanh())

    # Appearance Model forward
    def forward(self, inputs):
        # Convolution 3x3 kernel Layer 1 (3@36x36 -> 32@36x36)
        a1 = self.a_conv_layer1(inputs)
        # Convolution 3x3 kernel Layer 2 (32@36x36 -> 32@36x36)
        a2 = self.a_conv_layer2(a1)
        # Attention 1
        attention1 = AttentionBlock(a2)

        # Dropout 0.5
        a3 = self.a_dropout(a2)
        # Average-pooling 2x2 kernel Layer 3 (32@36x36 -> 32@18x18)
        a4 = self.avg_pool1(self.a_dropout(a3))

        # Convolution 3x3 kernel Layer 4 (32@18x18 -> 64@18x18)
        a5 = self.a_conv_layer3(a4)
        # Convolution 3x3 kernel Layer 5 (64@18x18 -> 64@18x18)
        a6 = self.a_conv_layer4(a5)
        # Attention 2
        attention2 = AttentionBlock(a6)

        return attention1, attention2
