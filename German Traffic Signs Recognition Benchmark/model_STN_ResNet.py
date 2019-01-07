# Author- Aishwarya Budhkar
# ResNet-STN model

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    # Forward propagation
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
#ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
	  #STN code for color variance
        self.conv6 = nn.Conv2d(3, 10, kernel_size=1, stride=1, padding=2)
        self.conv6_drop = nn.Dropout2d(0.05)
        self.ln_drop=nn.Dropout2d(0.6)
        self.bn6 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 3, kernel_size=1, stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d(0.05)
        self.bn2 = nn.BatchNorm2d(3)
	  #STN code for spatial variance
        self.conv3 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.l1 = nn.Linear(2304, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64,6 )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.avgpool = nn.AvgPool2d(2,stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    #Forward prop
    def forward(self, x):
	   #Leaky relu to avoid dying relu units
        out = self.bn6(F.leaky_relu(self.conv6_drop(self.conv6(x))))
        out = self.bn2(F.leaky_relu(self.conv2_drop(self.conv2(out))))
        out = F.max_pool2d(self.conv3(out), 2)
        out = F.max_pool2d(self.conv4(out), 2)
        out = F.max_pool2d(self.conv5(out), 2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
        out = self.ln_drop(self.l3(out))
        out = out.view(-1, 2, 3)
        grid = F.affine_grid(out, x.size())
        x1 = F.grid_sample(x, grid)
        out = self.maxpool(F.relu(self.bn1(self.conv1(x1))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out)	