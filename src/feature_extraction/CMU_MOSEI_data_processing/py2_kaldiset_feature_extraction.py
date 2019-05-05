#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: haoqi

# extract feature for cmu-mosei dataset

TASK: extract the whole session kaldi features


"""


import os, sys
import pandas as pd
sys.path.insert(0,'../extract_feat_funcs/')
from funcs_extract_featureset_kaldi import run



# load the csv file with dataset info
audio_dir = '/mnt/hdd2/haoqi_hdd2/data_set/CMU_MOSEI/Audio/Full/WAV_16000/'



data_info_csv = './Segments_File.csv'

df = pd.read_csv(data_info_csv)
meta_data_array_np = df.values

session_list = list(set(list(meta_data_array_np[:, 2])))
utterance_id_list = list(meta_data_array_np[:,1])

start_time_list = list(meta_data_array_np[:,3])
end_time_list = list(meta_data_array_np[:,4])

all_wav_file_list = [x+'.wav' for x in session_list]
output_sen_feat_dir = '/mnt/hdd2/haoqi_hdd2/data_after_processing/workspace_2018_ICASSP/CMU_MOSEI_feat/sentence_kaldi_featureset/'
os.system('mkdir -p '+output_sen_feat_dir)
# extract kaldi features
wav_list = [os.path.join(audio_dir, x) for x in all_wav_file_list]
for x in wav_list:
    if not os.path.exists(x):
        print (x)


run(wav_list, output_sen_feat_dir, normalize=True, num_mfcc=40, num_mfb=40, energy=True)   

