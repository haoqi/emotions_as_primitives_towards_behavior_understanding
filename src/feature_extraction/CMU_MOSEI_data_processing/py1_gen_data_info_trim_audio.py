#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:09:42 2018

CMU-MOSEI seg parser script

@author: haoqi
"""

import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from mmsdk import mmdatasdk
import pdb


def gen_start_end_segment(csd_file):

    #file_id_list=np.genfromtxt(idfile,dtype='str')
    mydict={'myfeatures':csd_file} 
    mydataset=mmdatasdk.mmdataset(mydict) 
    file_id_keys=list(mydataset.computational_sequences['myfeatures'].data.keys())

    utterance_ids=[]
    start_time=[]
    end_time=[]
    recording_ids=[]
    emo_dict = {}
    emotion_name_list = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
    for emo in emotion_name_list:
        emo_dict[emo] = []
    

    for file_id in tqdm(file_id_keys):
        interval_file_id=mydataset.computational_sequences['myfeatures'].data[file_id]['intervals']
        emotion_labels_lst = mydataset.computational_sequences['myfeatures'].data[file_id]['features']
     
        for i in np.arange(0,interval_file_id.shape[0]):

            start_time.append(interval_file_id[i][0])
            end_time.append(interval_file_id[i][1])
            utterance_id_create=file_id+"_"+str(interval_file_id[i][0])+"_"+str(interval_file_id[i][1])
            utterance_ids.append(utterance_id_create)
            recording_ids.append(file_id)
            
            # process the emotion labels
            for idx_emo,emo in enumerate(emotion_name_list):
                emo_dict[emo].append(emotion_labels_lst[i][idx_emo])
            #pdb.set_trace()
                
    segments_file = OrderedDict()
    segments_file['utterance_id']=utterance_ids
    segments_file['recording_id']=recording_ids
    segments_file['start_time']=start_time
    segments_file['end_time']=end_time
    
    #pdb.set_trace()
    
    for emo in emotion_name_list:
        segments_file[emo]=emo_dict[emo]
    
    
    segments_file=pd.DataFrame(segments_file)
    segments_file.to_csv('Segments_File.csv')


if __name__ =='__main__':
    csd_info_file = '/mnt/hdd2/haoqi_hdd2/cmu_data_sdk/cmumosei/CMU_MOSEI_LabelsEmotions.csd'
    gen_start_end_segment(csd_info_file)


