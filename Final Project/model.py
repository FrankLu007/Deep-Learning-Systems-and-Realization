import torch.nn as nn
import torchvision

class ImageNet(nn.Module):

    def __init__(self, OutputSize):
        super(ImageNet, self).__init__()
        self.ImageNet = torchvision.models.mobilenet_v2(pretrained = True, progress = True)
        self.fc = nn.Linear(1000, OutputSize)
        
    def forward(self, x):
        x = self.ImageNet(x)
        return self.fc(x)