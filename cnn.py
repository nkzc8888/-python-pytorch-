# -*- coding: utf-8 -*-
"""
@author: nkzc8888
"""

from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )
        
    def forward(self, x):
        
        #x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x
        

