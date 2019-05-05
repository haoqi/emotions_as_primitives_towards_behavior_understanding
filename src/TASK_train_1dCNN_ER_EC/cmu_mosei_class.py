
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 2018

checked on Nov 14th, based on the official split
num_all = num_train+num_dev+num_eval
@author: haoqi
"""

import os
import pandas as pd
import numpy as np
import random
from cmu_mosei_std_folds import standard_train_fold, standard_test_fold, standard_valid_fold
import pdb


class CMU_MOSEI_dataset():

    def __init__(self):
        
        #f {happiness, sadness, anger, fear, disgust, surprise} are annotated on a [0,3] Like scale for presence of emotion x: [0: no evidence of x, 1: weakly x, 2: x, 3: highly x]
        
        self.emo_dict = {'hap':0, 'sad':1, 'anger':2, 'fear':3, 'disgust':4, 'surprise':5}
        self.num2emodict = {0:'hap', 1:'sad', 2: 'anger', 3:'fear', 4:'disgust', 5:'surprise'} 
        
        # load dataset info csv file
        data_info_csv = '../../dir_audio_processing/CMU_MOSEI_data_processing/Segments_File.csv'
        df = pd.read_csv(data_info_csv)
        meta_data_array_np = df.values
        
        self.utterance_id_list = list(meta_data_array_np[:,1]) # utterance id list
        self.session_id_list = list(meta_data_array_np[:,2]) # session id list
        #pdb.set_trace()
        self.tgt_np = np.array(meta_data_array_np[:,5:11]).astype(np.float32) # np array of emotion score
        # do label score normalization [0,3] normalize to [-1,1]
        self.tgt_np = np.divide(self.tgt_np,3) * 2 -1
        
        self.training_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_train_fold]
        
        self.dev_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_valid_fold]
        
        self.test_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_test_fold]
        
    def gen_train_pair(self):
        '''
        return the list of tuple of train_utterance, label, e.g.  ('-3g5yACwYnA_4.84_14.052', array([0.6666667 , 0.6666667 , 0., 0., 0., 0.33333334], dtype=float32))

        '''
        return self.training_pair_npy_list       
    def gen_eval_pair(self):       
        return self.test_pair_npy_list
    def gen_dev_pair(self):
        return self.dev_pair_npy_list
    def gen_num2emo_dict(self):
        return self.num2emodict
    
    def gen_all_session_list(self):
        return self.session_id_list

class CMU_MOSEI_dataset_classification():
    '''
    This class is used in the second stage training of classification accu
    Notes: changes from CMU_MOSEI_dataset: the tgt is conver to binary labels from original ratings
    Based on their papers for original rating >0 the emotion label is regarded as 1
    if the original rating ==0, the emotion label is regarded as 0
    '''
    def __init__(self):
        
        #f {happiness, sadness, anger, fear, disgust, surprise} are annotated on a [0,3] Like scale for presence of emotion x: [0: no evidence of x, 1: weakly x, 2: x, 3: highly x]
        
        self.emo_dict = {'hap':0, 'sad':1, 'anger':2, 'fear':3, 'disgust':4, 'surprise':5}
        self.num2emodict = {0:'hap', 1:'sad', 2: 'anger', 3:'fear', 4:'disgust', 5:'surprise'} 
        
        # load dataset info csv file
        data_info_csv = '../../dir_audio_processing/CMU_MOSEI_data_processing/Segments_File.csv'
        df = pd.read_csv(data_info_csv)
        meta_data_array_np = df.values
        
        self.utterance_id_list = list(meta_data_array_np[:,1]) # utterance id list
        self.session_id_list = list(meta_data_array_np[:,2]) # session id list

        self.tgt_np = np.array(meta_data_array_np[:,5:11]) # np array of emotion score
        # set the label to 1 once the original rating is larger than 0        
        self.tgt_np[self.tgt_np>0]=1
        self.tgt_np = self.tgt_np.astype('int_')

        self.training_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_train_fold]
        
        self.dev_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_valid_fold]
        
        self.test_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_test_fold]
        
    def gen_train_pair(self):
        '''
        return the list of tuple of train_utterance, label, e.g.  ('-3g5yACwYnA_4.84_14.052', array([0,0,1,0,1], dtype=int32))

        '''
        return self.training_pair_npy_list       
    def gen_eval_pair(self):       
        return self.test_pair_npy_list
    def gen_dev_pair(self):
        return self.dev_pair_npy_list
    def gen_num2emo_dict(self):
        return self.num2emodict
    def gen_all_emos(self):
        return ['hap', 'sad', 'anger', 'fear', 'disgust', 'surprise']
    
    def count_imbalance_distribution(self, emo_idx):
        # return the number of class labels
        count1 = np.sum(self.tgt_np[:,emo_idx]==1)
        count0 = np.sum(self.tgt_np[:, emo_idx]==0)
        return count0, count1



class CMU_MOSEI_dataset_classification_upsampling():
    '''
    This class is used in the second stage training of classification accu
    Notes: changes from CMU_MOSEI_dataset: the tgt is conver to binary labels from original ratings
    Based on their papers for original rating >0 the emotion label is regarded as 1
    if the original rating ==0, the emotion label is regarded as 0
    
    
    Due to the data imbalance we upsample the under-represented class
    '''
    def __init__(self):
        
        #f {happiness, sadness, anger, fear, disgust, surprise} are annotated on a [0,3] Like scale for presence of emotion x: [0: no evidence of x, 1: weakly x, 2: x, 3: highly x]
        
        self.emo_dict = {'hap':0, 'sad':1, 'anger':2, 'fear':3, 'disgust':4, 'surprise':5}
        self.num2emodict = {0:'hap', 1:'sad', 2: 'anger', 3:'fear', 4:'disgust', 5:'surprise'} 
        
        # load dataset info csv file
        data_info_csv = '../../dir_audio_processing/CMU_MOSEI_data_processing/Segments_File.csv'
        df = pd.read_csv(data_info_csv)
        meta_data_array_np = df.values
        
        self.utterance_id_list = list(meta_data_array_np[:,1]) # utterance id list
        self.session_id_list = list(meta_data_array_np[:,2]) # session id list

        self.tgt_np = np.array(meta_data_array_np[:,5:11]) # np array of emotion score
        
        # set the label to 1 once the original rating is larger than 0        
        self.tgt_np[self.tgt_np>0]=1
        self.tgt_np = self.tgt_np.astype('int_')

        self.training_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_train_fold]
        
        self.dev_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_valid_fold]
        
        self.test_pair_npy_list = [(self.utterance_id_list[i], self.tgt_np[i]) for i in range(len(self.utterance_id_list)) 
                                        if self.session_id_list[i] in standard_test_fold]
        
    def gen_train_pair(self):
        '''
        return the list of tuple of train_utterance, label, e.g.  ('-3g5yACwYnA_4.84_14.052', array([0,0,1,0,1], dtype=int32))

        '''
        return self.training_pair_npy_list       
    def gen_eval_pair(self):       
        return self.test_pair_npy_list
    def gen_dev_pair(self):
        return self.dev_pair_npy_list
    def gen_num2emo_dict(self):
        return self.num2emodict
    def gen_all_emos(self):
        return ['hap', 'sad', 'anger', 'fear', 'disgust', 'surprise']
    
    def count_imbalance_distribution(self, emo_idx):
        # return the number of class labels
        count1 = np.sum(self.tgt_np[:,emo_idx]==1)
        count0 = np.sum(self.tgt_np[:, emo_idx]==0)
        return count0, count1
    
    def gen_balanced_train_pair(self, emo_idx):
        return self.upsampling_cmu_mosei(self.gen_train_pair(), emo_idx)
        
    def upsampling_cmu_mosei(self, original_training_list, emo_idx):
        ''' the original data set has imbalance issue, upsampling the under-represented class'''
        
        # count number
        count_dict = {'0':0,'1':0}
        sep_file_list_dict = {}
        for item in count_dict:
            sep_file_list_dict[item] =[]
            
        for item in original_training_list:
            label = item[1][emo_idx] # only check that behavior
            count_dict[str(label)] +=1
            sep_file_list_dict[str(label)].append(item)
            
        balanced_list = []
        
        max_value_key = max(count_dict, key=count_dict.get)
        max_value = count_dict[max_value_key]
        
        for item in count_dict.keys():
            tmp_emo_list = sep_file_list_dict[item][:]
            
            ratio = max_value/len(tmp_emo_list)
            while ratio > 1:
                balanced_list += tmp_emo_list
                ratio -= 1
                random.shuffle(tmp_emo_list)
            
            if ratio > 0:
                balanced_list += tmp_emo_list[:int(len(tmp_emo_list)*ratio)]
                
        return balanced_list
        

if __name__=='__main__':
    dataset = CMU_MOSEI_dataset_classification()
    pdb.set_trace()
    

    
