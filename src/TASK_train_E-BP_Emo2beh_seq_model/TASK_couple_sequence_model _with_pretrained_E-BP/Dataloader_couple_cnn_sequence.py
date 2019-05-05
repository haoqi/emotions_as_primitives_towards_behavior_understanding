#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:13:45 2019

@author: haoqi
"""

import os
import torch
import math

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb


class Dataloader_couple_cnn_seq(Dataset):
    def __init__(self, sess_list):
        
        self.couple_in_list = sess_list
    
        self.beh_list = ['acceptance', 'blame', 'positive', 'negative', 'sadness']
        
        # load all features in to memory
        self.whole_feat_dict = {}
        #self.feat_length_dict = {}
        for item in self.couple_in_list:
            self.whole_feat_dict[os.path.basename(item[0])]= np.load(item[0])

    def __len__(self):
        return len(self.couple_in_list)
    
    def __getitem__(self, idx):
        
        
        npy_file_path_str = self.couple_in_list[idx][0]
        label_dict_list = self.couple_in_list[idx][1]
        sess_name_key  = os.path.basename(npy_file_path_str).split('.npy')[0]
        # genearte training labels
        return_label, return_mask = self.gen_label_seq(os.path.basename(npy_file_path_str), label_dict_list)
        feat  = self.whole_feat_dict[os.path.basename(npy_file_path_str)]
        feat_seq = self.gen_feature_seq(feat)
        len_curr = feat_seq.shape[0]
        return (feat_seq, np.asarray(return_label, dtype=np.float32), np.asarray(return_mask, dtype=np.float32), np.asarray(len_curr, dtype=np.int_), sess_name_key)
        
        
    def gen_feature_seq(self, feat, win_sz=100):
        
        # split feature sequence to multiple parts wach with length of win_sz
        
        N = math.floor(feat.shape[0]/win_sz)
        input_np = feat[:win_sz*N,:]
        
        output_np  = np.reshape(input_np, (-1,  win_sz, input_np.shape[1]))
        
        return output_np.astype('float32')
    
    def gen_label_seq(self, npy_str, input_label_dict_list):
        '''
        if the beh exists in binary score dict, return the 0/1 value, if it is not in the dict, return the 
        2 class label based on the split dict
        '''
        #print(npy_str)

        return_beh_score = []
        return_beh_mask = []    
        for x in self.beh_list:
            if x in input_label_dict_list:
                return_beh_score.append(input_label_dict_list[x])
                return_beh_mask.append(1)
            else:
                # the session is not in binary list
                return_beh_score.append(2.0)
                return_beh_mask.append(0)          
        return return_beh_score, return_beh_mask
        
        
        