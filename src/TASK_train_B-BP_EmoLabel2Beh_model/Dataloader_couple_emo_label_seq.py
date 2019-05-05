#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 20:36:04 2019

@author: haoqi
"""

import os
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb


class Dataloader_couple_emotion_label_seq(Dataset):
    def __init__(self, sess_list):
        
        self.couple_in_list = sess_list
        self.emo_list = ['hap', 'sad', 'anger', 'fear', 'disgust', 'surprise']
    
        self.beh_list = ['acceptance', 'blame', 'positive', 'negative', 'sadness']
        
        # load all couple sess seq
        whole_sess_emo_labels_dict_npy = '/auto/rcf-proj3/pg/haoqili/workspace_2018_icassp/data_couples/emo_label_1s_from_CMU/'+\
                                        'couple_emotion_label_1s.npy'
        self.whole_sess_emo_labels_dict = np.load(whole_sess_emo_labels_dict_npy).item()
        
    def __len__(self):
        return len(self.couple_in_list)
    
    def __getitem__(self, idx):
        
        
        sess_name_key = self.couple_in_list[idx][0]
        label_dict_list = self.couple_in_list[idx][1]
        return_label, return_mask = self.gen_label_seq(sess_name_key+'.npy', label_dict_list)
        feat_emo_seq = self.gen_emo_feature_seq(sess_name_key)
        len_curr = feat_emo_seq.shape[0]
        
        return (feat_emo_seq, np.asarray(return_label, dtype=np.float32), np.asarray(return_mask, dtype=np.float32), np.asarray(len_curr, dtype=np.int_), sess_name_key)
        
        
    def gen_emo_feature_seq(self, sess_name):
        
        feat_list_list = [self.whole_sess_emo_labels_dict[sess_name][emo] for emo in self.emo_list]
        feat = np.asarray(feat_list_list, dtype=np.float32)
        feat_transpose = np.transpose(feat)
        
        return feat_transpose
        
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
        
        
        
