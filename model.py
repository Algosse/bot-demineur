import torch.nn as nn
import torch.nn.functional as F


class BotDemineur(nn.Module):
    
    def __init__(self, rows = 16, cols = 30, outputs = 480):
        
        super(BotDemineur, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(3)
        
        # Due to no padding, the size of the output of conv2 is w-2 x h-2
        linear_output_size = (rows - 2) * (cols - 2) * 3
        self.head = nn.Linear(linear_output_size, outputs)
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))