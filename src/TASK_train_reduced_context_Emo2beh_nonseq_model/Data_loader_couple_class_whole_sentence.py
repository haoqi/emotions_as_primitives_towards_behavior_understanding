
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:35:41 2018

@author: haoqi

"""
import os
import torch
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb


class Couple_datasest_torch_train_2_classes_whole_session(Dataset):
    
    def __init__(self, input_file_list, middle_split_dict_str, flag_load_all_feat_into_mem=True):
        '''
        # the input_file_list is from couple_data_info class
        # [(npy name str, dict of available emotion labels)]
        
        # ouput for each behavior ouput has 3 values: 0, 1, 2
        # 0, 1 is identical to the original label, 2 is the label of the middle will be masked out
        '''
        self.couple_in_list = input_file_list
        self.beh_list = ['acceptance', 'blame', 'positive', 'negative', 'sadness']

        
        self.middle_split_dict_str = middle_split_dict_str
        self.middle_split_dict = np.load(self.middle_split_dict_str).item()
        
        self.flag_load_all_feat_into_mem = flag_load_all_feat_into_mem 
        
        # load all features in to memory
        if self.flag_load_all_feat_into_mem:
            self.whole_feat_dict = {}
            for item in self.couple_in_list:
                self.whole_feat_dict[os.path.basename(item[0])]= np.load(item[0])
                

             
    def __len__(self):
        return len(self.couple_in_list)
    
    def __getitem__(self, idx):
        '''For each session, randomly select one sample'''

        npy_file_str = self.couple_in_list[idx][0]
        label_dict_list = self.couple_in_list[idx][1]
        feat_file_name = os.path.basename(npy_file_str)
        
        
        return_label, return_mask = self.gen_label_seq(os.path.basename(npy_file_str),label_dict_list)
        
        if self.flag_load_all_feat_into_mem:
            # all feature is already in the memory
            feat = self.whole_feat_dict[os.path.basename(npy_file_str)]
        else:
            feat  = np.load(npy_file_str)
        
        len_curr = feat.shape[0]
    
        return (feat.astype('float32'), \
                    np.asarray(return_label, dtype=np.float32), np.asarray(return_mask, dtype=np.float32), np.asarray(len_curr, dtype=np.int_), feat_file_name)
        
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
    
