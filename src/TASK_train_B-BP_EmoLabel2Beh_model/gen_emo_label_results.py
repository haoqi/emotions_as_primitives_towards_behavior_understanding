#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:32:21 2019

@author: haoqi


based on pretrained cmu-emotion model, test on couple dataset

"""

import os, sys
import glob
import numpy as np
import math


import torch
import pdb



sys.path.insert(0, '/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/src/TASK_train_1dCNN_cmu_kaldi_featureset/')

from model_cnn1d_cmu_classification_v1 import Classification_Base_1D_NN_fixed_seq_len_1s_majvote_v2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_emo_label(emo, in_data, pretrained_emotion_ckpt):
    '''
    load the pretrained emo model and test on couple data
    
    '''
    global device
    input_channels = 84 # feature dimension is 84
    model_emo  = Classification_Base_1D_NN_fixed_seq_len_1s_majvote_v2(input_channels)
    saved_ckpt_pth = pretrained_emotion_ckpt[emo]
    model_emo.load_state_dict(torch.load(saved_ckpt_pth)['state_dict'])
    model_emo.eval()
    model_emo.to(device)
    
    in_data_tensor = torch.from_numpy(in_data)
    N  = math.floor(in_data.shape[0] / 100 ) # 100 samples per second
    input_tensor = in_data_tensor[:100* N,: ].view(-1, 100, 84)
    
    input_tensor = input_tensor.permute(0, 2,1).to(device)
    
    output = model_emo(input_tensor)
    
    max_value, labels = torch.max(output, dim=1)
    return labels.tolist()

emotion_list =  ['hap', 'sad', 'anger', 'fear', 'disgust', 'surprise']
emotion_ckpt_root = '/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/src/TASK_train_1dCNN_cmu_kaldi_featureset/ckpts_cnn1d_cmumosei_classification/'

pretrained_emotion_ckpt_dict = {}
pretrained_emotion_ckpt_dict['hap']      = os.path.join(emotion_ckpt_root,'try1_weight_classification_balanced_batch32','hap_epoch_10_best_accu0.650_f1_0.668.pth')
pretrained_emotion_ckpt_dict['sad']      = os.path.join(emotion_ckpt_root,'try1_weight_classification_balanced_batch32','sad_epoch_28_best_accu0.620_f1_0.702.pth')
pretrained_emotion_ckpt_dict['anger']    = os.path.join(emotion_ckpt_root,'try1_weight_classification_balanced_batch128','anger_epoch_40_best_accu0.622_f1_0.782.pth')
pretrained_emotion_ckpt_dict['fear']     = os.path.join(emotion_ckpt_root,'try1_weight_classification_balanced_batch128','fear_epoch_48_best_accu0.563_f1_0.845.pth')
pretrained_emotion_ckpt_dict['disgust']  = os.path.join(emotion_ckpt_root,'try1_weight_classification_balanced_batch128','disgust_epoch_66_best_accu0.655_f1_0.846.pth')
pretrained_emotion_ckpt_dict['surprise'] = os.path.join(emotion_ckpt_root,'try1_weight_classification_balanced_batch32','surprise_epoch_66_best_accu0.580_f1_0.638.pth')

#----------------------
# all couple file_list
#----------------------
all_couple_np_list = glob.glob(os.path.join('/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/data_couples/kaldi_featureset', '*.npy'))
output_dir = os.path.join('/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/data_couples/', 'emo_label_1s_from_CMU')
ouput_value_dict={}
len_list= []
for npy in all_couple_np_list:
    npy_name = os.path.basename(npy).split('.npy')[0]
    print(npy_name)
    ouput_value_dict[npy_name] = {}
    feat = np.load(npy)
    for emo in emotion_list:
        predict_labels = test_emo_label(emo , feat, pretrained_emotion_ckpt_dict)
        ouput_value_dict[npy_name][emo] = predict_labels
    len_list.append(len(predict_labels))
    if len(predict_labels)==543:
        print('------------------------------------------------')
        
        print(npy)
        print('------------------------------------------------')
pdb.set_trace()
# save to file
np.save(os.path.join(output_dir, 'couple_emotion_label_1s.npy'), ouput_value_dict)

