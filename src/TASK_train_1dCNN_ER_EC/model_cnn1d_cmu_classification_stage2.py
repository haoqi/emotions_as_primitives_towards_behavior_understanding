#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:10:20 2018

@author: haoqi
"""
import os
import torch

import torch.nn as nn


class Classification_Base_1D_NN_fixed_seq_len_1s_majvote_v2(nn.Module):
    '''
    '''
    def __init__(self, in_channels_num):
        super(Classification_Base_1D_NN_fixed_seq_len_1s_majvote_v2, self).__init__()
        self.num_out_beh = 6
        self.cnn_1d=nn.Sequential(
                nn.Conv1d(in_channels_num, 96, kernel_size=10, stride=2, padding=0), # input_len=1*100, out 46
                nn.ReLU(),
                nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 46, out 21
                nn.ReLU(),
                nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 21, out 9  
                nn.ReLU(),
                nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),# in 9, out 4  
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
                )
        self.out1 = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
                )
        self.out2 = nn.Sequential(
                nn.Linear(128,64),
                nn.PReLU(),
                nn.Linear(64, 64),
                nn.PReLU(),
                nn.Linear(64, 2)
                )
    def forward(self, x_input):
        x = self.cnn_1d(x_input)
        x = x.view(x.size(0), -1)
        output1 = self.out1(x)
        output = self.out2(output1)
        return output
