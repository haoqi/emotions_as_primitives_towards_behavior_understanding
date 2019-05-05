#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:53:15 2019

@author: haoqi

RNN model for emotion list to beh score

"""

import torch
import torch.nn as nn

import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Emotion_Seq2Beh_Model(nn.Module):
    def __init__(self):
        super(Emotion_Seq2Beh_Model, self).__init__()
        self.num_of_emotions = 6
        self.num_of_behs = 5
        self.hidden_sz = 128
        self.num_layers = 2
        
        self.gru1 = nn.GRU(input_size=self.num_of_emotions, hidden_size=self.hidden_sz, num_layers=self.num_layers)
        
        self.fc_out = nn.Sequential(
                nn.Linear(self.hidden_sz, int(self.hidden_sz/2)),
                nn.ReLU(),
                nn.Linear(int(self.hidden_sz/2), self.num_of_behs)
                )
        
    def forward(self, x_input):
        batch_sz = x_input.shape[1]

        h_init = self.initHidden(batch_sz)
        x, hidden = self.gru1(x_input, h_init)
        x_ouput = self.fc_out(x[-1,:,:]) # only return last time step's results, this is a many-to-one sequence problem
        return x_ouput
        
    def initHidden(self, batch_sz):
        return torch.zeros(self.num_layers, batch_sz, self.hidden_sz, device=device)
        

        







