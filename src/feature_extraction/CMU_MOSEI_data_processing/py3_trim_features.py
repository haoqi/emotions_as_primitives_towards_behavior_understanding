#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:33:48 2018

the kaldi feature set is already extracted at py02*.py

TASK: trim the utterance feature from the cmu-mosei session




@author: haoqi
"""


import os
import numpy as np
import pandas as pd
import pdb

# load the feature info csv file


data_info_csv = './Segments_File.csv'

df = pd.read_csv(data_info_csv)
meta_data_array_np = df.values

session_list = list(set(list(meta_data_array_np[:, 2])))
utterance_id_list = list(meta_data_array_np[:,1])
start_time_list = list(meta_data_array_np[:,3])
end_time_list = list(meta_data_array_np[:,4])



session_sen_feat_dir = '/mnt/hdd2/haoqi_hdd2/data_after_processing/workspace_2018_ICASSP/CMU_MOSEI_feat/sentence_kaldi_featureset/'
output_utterance_feat_dir = '/mnt/hdd2/haoqi_hdd2/data_after_processing/workspace_2018_ICASSP/CMU_MOSEI_feat/trimmed_utterance_kaldi_featureset/'
os.system('mkdir -p '+output_utterance_feat_dir)


for i, utterance in enumerate( utterance_id_list):
    sess_name =  '_'.join(utterance.split('_')[:-2])+'.npy'
    #print(sess_name)
    
    start_time_split_value = float(utterance.split('_')[-2])
    end_time_split_value = float(utterance.split('_')[-1]) 
    
    start_time = start_time_list[i]
    end_time = end_time_list[i]
    
    
    if abs(start_time - start_time_split_value)<0.01 and abs(end_time - end_time_split_value)<0.01:
        feat_tmp = np.load(os.path.join(session_sen_feat_dir,sess_name))
        
        feat_current_frame = feat_tmp[round(start_time_split_value*100):round(end_time_split_value*100),:]
        
        np.save(os.path.join(output_utterance_feat_dir, utterance+'.npy'), feat_current_frame)
        del feat_current_frame, feat_tmp
    else:
        print('Time mismatch...')
        print(sess_name)
    
    
    
    





