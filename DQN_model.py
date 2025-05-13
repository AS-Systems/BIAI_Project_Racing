import torch
import torch.nn as nn
from config import FRAME_STACK

#source https://www.researchgate.net/figure/A-standard-DQN-architecture-with-convolutional-layers-allows-comparisons-between-buffer_fig4_321962850

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=4, stride=2)
        self.canv2 = nn.Conv2d(32,64,4,2)
        self.canv3 = nn.Conv2d(64,64,3)
        self.fc1 = nn.Linear(64*7*7,512)
        self.fc2 = nn.Linear(512,action_dim)
        
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.canv2(x))
        x = torch.relu(self.canv3(x))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        
        return self.fc2(x)