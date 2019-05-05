#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:09:35 2018

@author: haoqi

Eval script on train_supervised_2classes_binary_classification_sep_dev_whole_session_No7

the training is leave 4 out

"""


import os
import numpy as np
import pdb
import glob

couple_meta_root_dir = '/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/src/TASK_couple_train_1d_cnn_kaldi_featureset/couple_meta_data/'
unique_couple_id_file = os.path.join(couple_meta_root_dir, 'unique_couple_id.txt')
binary_ref_dict = '/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/data_couples/test_code/'

beh_list = ['acceptance', 'blame', 'positive', 'negative', 'sadness']

save_model_dir = '/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/src/TASK_couple_sequence_model/ckpts_couple_seq_emotion_pretrained_binary_trainNo2/'
expName = 'exp_model_cnn_seq_pre_N4'

dict_gender ={'W':'wife', 'H': 'hus'}

#print ('----epoch '+given_epoch + '----')
all_sub_dirs = glob.glob(os.path.join(save_model_dir, expName, '*/test_best_result.npy'))

for idx, beh in enumerate(beh_list):
    beh_binary_file = os.path.join(binary_ref_dict, beh+'.txt')
    with open(beh_binary_file, 'r') as f:
        file_list = f.read().splitlines()
    count_correct = 0
    totoal_count = 0
    for item in file_list:
        #'W,0,1.000,473.pre.ps.h'
        tmp_split  = item.split(',')
        gender = dict_gender[tmp_split[0]]
        ref_label = tmp_split[1]
        coupldID =  str(tmp_split[3].split('.')[0])

            
        for dir_ckpt in all_sub_dirs:
            if coupldID in dir_ckpt.split('/')[-2]:
                file_npy_best = dir_ckpt
                #print(coupldID)
                #print(file_npy_best)
                if os.path.exists(file_npy_best):
                    
                    #if os.path.isfile(file_npy_best):
                    results = np.load(file_npy_best).item()
                    key_dict = gender + '.' +tmp_split[3]
                    #print(key_dict)
                    #print(file_npy_best)
                    #print('--')
                    test_label = str(results[key_dict][idx])
                    
                    if ref_label == test_label:
                        count_correct +=1
                    totoal_count += 1

    print(beh)
    print(count_correct/totoal_count)
    print(totoal_count)
    
    
    
    
    
