#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:00:42 2019

@author: haoqi
"""


import torch
import torch.nn as nn
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the model is based on CMU emotion
# Base_1D_NN_fixed_seq_len_1s_majvote_v2

class model_cnn_seq_fully(nn.Module):
    def __init__(self, in_channels_num):
        super(model_cnn_seq_fully, self).__init__()
        
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
        self.num_of_behs = 5
        self.hidden_sz = 128
        self.num_layers = 2
        
        self.gru1 = nn.GRU(input_size=128, hidden_size=self.hidden_sz, num_layers=self.num_layers)
        
        self.fc_out = nn.Sequential(
                nn.Linear(self.hidden_sz, int(self.hidden_sz/2)),
                nn.ReLU(),
                nn.Linear(int(self.hidden_sz/2), self.num_of_behs)
                )
                
    def forward(self, x_input):

        x_cnn_output = self.cnn_1d(x_input)
        x = x_cnn_output.view(x_cnn_output.size(0), -1) # the shape of output from CNN is (Num_seq, channels=128)
        x_input_to_seq = torch.unsqueeze(x, 1)
        # change to the input of seq model (seq_len, batch, input_size):
        batch_sz = x_input_to_seq.shape[1]
        h_init = self.initHidden(batch_sz)
        x, hidden = self.gru1(x_input_to_seq , h_init)
        x_ouput = self.fc_out(x[-1,:,:]) # only return last time step's results, this is a many-to-one sequence problem
        return x_ouput
        
    def initHidden(self, batch_sz):
        return torch.zeros(self.num_layers, batch_sz, self.hidden_sz, device=device)
            
