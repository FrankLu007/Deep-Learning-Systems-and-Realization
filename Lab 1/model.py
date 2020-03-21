import torch.nn as nn
import torchvision

class ImageNet(nn.Module):


    def __init__(self):
        super(ImageNet, self).__init__()
        self.ImageNet = torchvision.models.wide_resnet50_2(pretrained = True)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 11)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.ImageNet(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x