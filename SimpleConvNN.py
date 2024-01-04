import torch


class SimpleConvNN(torch.nn.Module):
    def __init__(self):
        super(SimpleConvNN, self).__init__()
        # 50 x 62 x 3
        self.conv1 = torch.nn.Conv2d(3, 32, 3)  # 48 x 60 x 32
        # MaxPool 2x2 24 x 30 x 32
        self.conv2 = torch.nn.Conv2d(32, 64, 3)  # 22 x 28 x 64
        # MaxPool 2x2 11 x 14 x 64
        self.conv3 = torch.nn.Conv2d(64, 64, 3)  # 9 x 12 x 64
        # MaxPool 2x2 4 x 6 x 64
        self.fc = torch.nn.Linear(4 * 6 * 64, 7)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
