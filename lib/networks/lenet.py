import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LeNet(nn.Module):

    def __init__(self, num_classes=10, rp_1=False, rp1_out_channel=0, rp_2=False, rp2_out_channel=0):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.rp_1 =rp_1
        self.rp_2 =rp_2

        if rp_1:
            self.rp1 = nn.Conv2d(in_channels=3, out_channels=rp1_out_channel, kernel_size=5, stride=1)
            self.rp_conv1 = nn.Conv2d(in_channels=3, out_channels= 64-rp1_out_channel, kernel_size=5, stride=1)
            self.rp1.weight.requires_grad = False
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        if rp_2:
            self.rp2 = nn.Conv2d(in_channels=64, out_channels=rp2_out_channel, kernel_size=5, stride=1)
            self.rp_conv2 = nn.Conv2d(in_channels=64, out_channels=128-rp2_out_channel, kernel_size=5, stride=1)
            self.rp2.weight.requires_grad = False
        else:
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)  # 5*5 from image dimension
        self.fc2 = nn.Linear(1024, 1024)
        
        self.fc3 = nn.Linear(1024, num_classes)
    
    def random_rp_matrix(self):
        weight = None
        for name, param in self.named_parameters():
            if ('rp' in name and 'weight' in name) and 'conv' not in name:
                kernel_size = torch.zeros_like(param.data)
                nn.init.kaiming_normal_(kernel_size, nonlinearity='conv2d')
                param.data = kernel_size.detach()
                weight = kernel_size.detach()
            elif ('rp' in name and 'bias' in name) and 'conv' not in name:
                new_bias = torch.zeros_like(param.data)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(new_bias, -bound, bound)
                param.data = new_bias.detach()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        if self.rp_1:
            out1 = torch.cat([self.rp_conv1(x), self.rp1(x)], dim=1)
            x = F.max_pool2d(F.relu(out1), (2, 2))
        else:
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        if self.rp_2:
            out2 = torch.cat([self.rp_conv2(x), self.rp2(x)], dim=1)
            x = F.max_pool2d(F.relu(out2), 2)
        else:
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

