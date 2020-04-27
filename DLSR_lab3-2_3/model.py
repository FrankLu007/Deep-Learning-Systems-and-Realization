import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, InputCahnnel, width):
        super(Block, self).__init__()
        self.CNN1 = nn.Conv2d(InputCahnnel, width, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.BN1 = nn.BatchNorm2d(width)
        self.CNN2 = nn.Conv2d(width, width, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.BN2 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace = True)
        # self.parameter = nn.Parameter(torch.zeros(1), requires_grad = True)

    def forward(self, InputData):
        data = self.BN2(self.CNN2(self.relu(self.BN1(self.CNN1(InputData)))))
        if data.shape == InputData.shape:
            data += InputData
        return self.relu(data)

def MakeLayer(width, depth):
    BlockList = [Block(int(width / 2), width)] + [Block(width, width) for _ in range(depth - 1)]
    return nn.Sequential(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1), *BlockList)

class ReZero(nn.Module):
    def __init__(self, depth, width, resolution, OutputSize = 11):
        super(ReZero, self).__init__()
        self.CNN_Start = nn.Sequential(
            nn.Conv2d(3, int(32 * width), kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(int(32 * width))
            )
        self.relu = nn.ReLU(inplace = True)
        self.CNN_Layers = nn.Sequential(MakeLayer(int(64 * width), depth), MakeLayer(int(128 * width), depth), MakeLayer(int(256 * width), depth), MakeLayer(int(512 * width), depth))
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        FC_InputSize = int(512 * width)
        self.FC = nn.Linear(FC_InputSize, OutputSize)

    def forward(self, InputData):
        data = self.relu(self.CNN_Start(InputData))
        data = self.CNN_Layers(data)
        data = self.AvgPool(data)
        data = torch.flatten(data, 1)
        # print(data.shape)
        return self.FC(data)