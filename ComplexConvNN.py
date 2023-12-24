import torch
from torch import nn
from torch.nn.functional import relu


class ComplexConvNN(nn.Module):
    def __init__(self):
        super(ComplexConvNN, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        # 100 x 125 x 3
        self.conv1 = nn.Conv2d(3, 32, 3) # 98 x 123 x 32
        self.batchnorm1 = nn.BatchNorm2d(32)
        # MaxPool 2x2 49 x 61 x 32
        self.conv2 = nn.Conv2d(32, 64, 3) # 47 x 59 x 64
        self.batchnorm2 = nn.BatchNorm2d(64)
        # MaxPool 2x2 23 x 29 x 64
        self.conv3 = nn.Conv2d(64, 128, 3)# 21 x 27 x 128
        self.batchnorm3 = nn.BatchNorm2d(128)
        # MaxPool 2x2 10 x 13 x 128
        self.conv4 = nn.Conv2d(128, 256, 3) # 8 x 11 x 256
        self.batchnorm4 = nn.BatchNorm2d(256)
        # MaxPool 2x2 4 x 5 x 256
        self.conv5 = nn.Conv2d(256, 512, 3) # 2 x 3 x 512
        self.batchnorm5 = nn.BatchNorm2d(512)
        # MaxPool 2x2 1 x 1 x 512
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        x = self.maxpool(self.batchnorm1(relu(self.conv1(x))))
        x = self.maxpool(self.batchnorm2(relu(self.conv2(x))))
        x = self.maxpool(self.batchnorm3(relu(self.conv3(x))))
        x = self.maxpool(self.batchnorm4(relu(self.conv4(x))))
        x = self.maxpool(self.batchnorm5(relu(self.conv5(x))))
        x = nn.functional.adaptive_avg_pool2d(x,(1,1))
        x = self.flatten(x)
        x = self.fc(x)
        return x