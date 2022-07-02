from torch.nn import Module
from sub_models.MotionModel import MotionModel_2D
from sub_models.AppearanceModel import AppearanceModel_2D
from sub_models.LinearModel import LinearModel


class DeepPhys(Module):
    def __init__(self):
        super.__init__()
        self.in_channel = 3
        self.out_channel = 32
        self.kernel_size = 3

        self.appearance_model = AppearanceModel_2D(self.in_channel, self.out_channel, self.kernel_size)
        self.motion_model = MotionModel_2D(self.in_channel, self.out_channel, self.kernel_size)

    def forward(self, inputs):
        m_model = self.motion_model.forward(inputs, self.appearance_model(inputs))
