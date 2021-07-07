#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:16:44 2021

@author: kratochvila
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MY_LeNetBN(BaseNet):

    def __init__(self, rep_dim: int=1000, in_channels: int=3, in_size=32):
        super().__init__()

        self.rep_dim = rep_dim#1000#128
        self.out_size = int(in_size/8)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels, 32, 5, bias=True, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=True, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=True, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * self.out_size * self.out_size, self.rep_dim, bias=True) # 4 * 4

        nn.init.xavier_normal_(self.conv1.weight, gain=1.0)
        nn.init.xavier_normal_(self.conv2.weight, gain=1.0)
        nn.init.xavier_normal_(self.conv3.weight, gain=1.0)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class MY_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128#128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=True, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=True, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=True, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 5, bias=True, padding=2)
        self.fc1 = nn.Linear(256 * 2 * 2, self.rep_dim, bias=True) # 20 * 15

        #self.soft = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1.0)
        nn.init.xavier_normal_(self.conv2.weight, gain=1.0)
        nn.init.xavier_normal_(self.conv3.weight, gain=1.0)
        nn.init.xavier_normal_(self.conv4.weight, gain=1.0)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(x))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(x))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(x))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = self.soft(x)
        return x
