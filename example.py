from ConfigurableNet import ConfigurableNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from ray import tune


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    config = {
            "lr": tune.grid_search([0.001, 0.1]),
            'lrBench' : tune.grid_search([{'lrPolicy': 'SINEXP', 'k0': 1.0, 'k1':3.0, 'l': 5, 'gamma':0.94},
                                  {'lrPolicy': 'POLY', 'k0': 0.2, 'k1':1.0, 'p':1.2, 'l':30},
                                  {'lrPolicy': 'FIX'}]),
            'stop_iteration': 200
        }
    convnet = ConvNet()
    Net = ConfigurableNet()
    Net.set_searchspace(torch, convnet, torch.utils.data.DataLoader, config, torch.optim, torchvision)
    Net.data_loader("~/data")
    Net.run()
