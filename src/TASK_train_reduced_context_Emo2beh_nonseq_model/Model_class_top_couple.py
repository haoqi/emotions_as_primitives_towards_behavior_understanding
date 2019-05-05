#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:12:59 2018

@author: haoqi
"""


import os
import torch

import torch.nn as nn
import pdb


class Couple_top_1d_NN_v1_input_full_session_binary_classification(nn.Module):
    def __init__(self, in_channels_num):
        super(Couple_top_1d_NN_v1_input_full_session_binary_classification, self).__init__()
        self.num_out_beh = 5 
        
        # pretrained model
        self.cnn_1d=nn.Sequential(
            nn.Conv1d(in_channels_num, 96, kernel_size=10, stride=2, padding=0), # input_len=1*100, out 46
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 46, out 21
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 21, out 9  
            nn.ReLU()
            #nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),# in 9, out 4  
            #nn.ReLU()
            #nn.AdaptiveMaxPool1d(1)
            )  
        self.cnn_1d_top=nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1)
            )        
        self.out1_top = nn.Sequential(
                nn.Linear(128, 128),
                #nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_out_beh)
                )

    def forward(self, x_input, len_mask):
        len_mask_list = len_mask.tolist()
        x = torch.cat([self.cnn_1d_top(self.cnn_1d(torch.unsqueeze(x_input[i][:, :len_mask_list[i]], 0))) for i in range(x_input.shape[0])],0)
        x = x.view(x.size(0), -1)
        output = self.out1_top(x)

        return output

class Couple_top_1d_NN_v1_input_full_session_binary_classification_rm1pool(nn.Module):
    def __init__(self, in_channels_num):
        super(Couple_top_1d_NN_v1_input_full_session_binary_classification_rm1pool, self).__init__()
        self.num_out_beh = 5 
        
        # pretrained model
        self.cnn_1d=nn.Sequential(
            nn.Conv1d(in_channels_num, 96, kernel_size=10, stride=2, padding=0), # input_len=1*100, out 46
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 46, out 21
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 21, out 9  
            nn.ReLU()
            #nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),# in 9, out 4  
            #nn.ReLU()
            #nn.AdaptiveMaxPool1d(1)
            )  
        self.cnn_1d_top=nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            #nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1)
            )        
        self.out1_top = nn.Sequential(
                nn.Linear(128, 128),
                #nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_out_beh)
                )

    def forward(self, x_input, len_mask):
        len_mask_list = len_mask.tolist()
        x = torch.cat([self.cnn_1d_top(self.cnn_1d(torch.unsqueeze(x_input[i][:, :len_mask_list[i]], 0))) for i in range(x_input.shape[0])],0)
        x = x.view(x.size(0), -1)
        output = self.out1_top(x)

        return output


class Couple_top_1d_NN_v1_input_full_session_binary_classification_rm2pool(nn.Module):
    def __init__(self, in_channels_num):
        super(Couple_top_1d_NN_v1_input_full_session_binary_classification_rm2pool, self).__init__()
        self.num_out_beh = 5 
        
        # pretrained model
        self.cnn_1d=nn.Sequential(
            nn.Conv1d(in_channels_num, 96, kernel_size=10, stride=2, padding=0), # input_len=1*100, out 46
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 46, out 21
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 21, out 9  
            nn.ReLU()
            #nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),# in 9, out 4  
            #nn.ReLU()
            #nn.AdaptiveMaxPool1d(1)
            )  
        self.cnn_1d_top=nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            #nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            #nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1)
            )        
        self.out1_top = nn.Sequential(
                nn.Linear(128, 128),
                #nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_out_beh)
                )

    def forward(self, x_input, len_mask):
        len_mask_list = len_mask.tolist()
        x = torch.cat([self.cnn_1d_top(self.cnn_1d(torch.unsqueeze(x_input[i][:, :len_mask_list[i]], 0))) for i in range(x_input.shape[0])],0)
        x = x.view(x.size(0), -1)
        output = self.out1_top(x)

        return output

class Couple_top_1d_NN_v1_input_full_session_binary_classification_add1pool(nn.Module):
    def __init__(self, in_channels_num):
        super(Couple_top_1d_NN_v1_input_full_session_binary_classification_add1pool, self).__init__()
        self.num_out_beh = 5 
        
        # pretrained model
        self.cnn_1d=nn.Sequential(
            nn.Conv1d(in_channels_num, 96, kernel_size=10, stride=2, padding=0), # input_len=1*100, out 46
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 46, out 21
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 21, out 9  
            nn.ReLU()
            #nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),# in 9, out 4  
            #nn.ReLU()
            #nn.AdaptiveMaxPool1d(1)
            )  
        self.cnn_1d_top=nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0),  
            nn.AvgPool1d(2, stride=2), # newly added
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1)
            )        
        self.out1_top = nn.Sequential(
                nn.Linear(128, 128),
                #nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_out_beh)
                )

    def forward(self, x_input, len_mask):
        len_mask_list = len_mask.tolist()
        x = torch.cat([self.cnn_1d_top(self.cnn_1d(torch.unsqueeze(x_input[i][:, :len_mask_list[i]], 0))) for i in range(x_input.shape[0])],0)
        x = x.view(x.size(0), -1)
        output = self.out1_top(x)

        return output


class Couple_top_1d_NN_v1_input_full_session_binary_classification_add2pool(nn.Module):
    def __init__(self, in_channels_num):
        super(Couple_top_1d_NN_v1_input_full_session_binary_classification_add2pool, self).__init__()
        self.num_out_beh = 5 
        
        # pretrained model
        self.cnn_1d=nn.Sequential(
            nn.Conv1d(in_channels_num, 96, kernel_size=10, stride=2, padding=0), # input_len=1*100, out 46
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 46, out 21
            nn.ReLU(),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), # in 21, out 9  
            nn.ReLU()
            #nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),# in 9, out 4  
            #nn.ReLU()
            #nn.AdaptiveMaxPool1d(1)
            )  
        self.cnn_1d_top=nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0), 
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=0),  
            nn.AvgPool1d(2, stride=2), # newly added
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=0),
            nn.AvgPool1d(2, stride=2), # newly added
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1)
            )        
        self.out1_top = nn.Sequential(
                nn.Linear(128, 128),
                #nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_out_beh)
                )

    def forward(self, x_input, len_mask):
        len_mask_list = len_mask.tolist()
        x = torch.cat([self.cnn_1d_top(self.cnn_1d(torch.unsqueeze(x_input[i][:, :len_mask_list[i]], 0))) for i in range(x_input.shape[0])],0)
        x = x.view(x.size(0), -1)
        output = self.out1_top(x)

        return output

