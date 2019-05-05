#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:22:44 2019

@author: haoqi
"""

import os
import random
random.seed(1234)
import numpy as np
#---------------------------------------------
# select dev and given test
        
class Couple_info_class_binary_w_mask_split_dev():
    '''
    
    '''
    def __init__(self, given_test_list=[], num_dev_couple=5):
        
        root_dir = '/auto/rcf-proj3/pg/haoqili/workspace_2018_icassp/src/TASK_couple_train_1d_cnn_kaldi_featureset/'
        # 372 file list
        file_list_372 = os.path.join(root_dir,'./couple_meta_data/372_file_list.txt')
        
        # all mean score list, from sandeep, in this case, will not use it
        all_mean_score_npy = os.path.join(root_dir,'./couple_meta_data/dict_couple_eval_mean_score_correct.npy')
        
        # binary label of all 372 session
        binary_score_npy = os.path.join(root_dir, './couple_meta_data/372_file_binary_label.npy')
        
        # couple data feature root dir
        self.feat_npy_dir = '/auto/rcf-proj3/pg/haoqili/workspace_2018_icassp/data_couples/kaldi_featureset/'
    
        # behavior list
        self.beh_list = ['acceptance', 'blame', 'positive', 'negative', 'sadness']
        self.dev_couple_id = []
        self.couple_id_set = [] # set of unique couple ID
        
        self.session_train_list_pair, self.session_dev_list_pair, self.session_test_list_pair=\
                self.load_couple_info(all_mean_score_npy, binary_score_npy, file_list_372, given_test_list, num_dev_couple)
        
    
    def load_couple_info(self, rating_info_npy, binary_label_info_npy,\
                         input_file_list_txt, given_test_list, num_dev_couple):
        # load info from pre-saved npy file 
        with open(input_file_list_txt, 'r') as f_in:
            file_list = f_in.read().splitlines()
        
        # 105.26wk.ps.h sample string
        all_unique_couple_id_lst = [x.split('.')[0] for x in file_list]
        all_unique_couple_id_lst = list(set(all_unique_couple_id_lst)) # 104 different couple ids
        self.couple_id_set = all_unique_couple_id_lst[:]
        
        # load mean rating and label npy dict
        all_couple_rating_dict = np.load(rating_info_npy).item()
        binary_score_dict = np.load(binary_label_info_npy).item()
        
        # split dev and training couple ID
        if len(given_test_list) == 0:
            # NEED to check this branch
            # randomly selected couple for dev and train
            random.shuffle(all_unique_couple_id_lst)
            dev_couple_id_lst = all_unique_couple_id_lst[0:num_dev_couple]
            train_couple_id_lst = all_unique_couple_id_lst[num_dev_couple:]
            self.dev_couple_id = dev_couple_id_lst
        else:
            # use the given test id as test id
            self.test_couple_id = given_test_list
            test_couple_id_lst = given_test_list
            
            train_dev_couple_id_lst = [x for x in all_unique_couple_id_lst if x not in self.test_couple_id]
            
            random.shuffle(train_dev_couple_id_lst)
            dev_couple_id_lst = train_dev_couple_id_lst[0:num_dev_couple]
            self.dev_couple_id = dev_couple_id_lst[:]
            train_couple_id_lst = train_dev_couple_id_lst[num_dev_couple:]
            
        # generate training and dev couple npy list
        session_train_list = []
        session_dev_list = []
        session_test_list = []
        
        for file in file_list:
            couple_id = file.split('.')[0]
            if couple_id in train_couple_id_lst:
                session_train_list.append('hus.'+file)#+'_merged.npy')
                session_train_list.append('wife.'+file)#+'_merged.npy')
            if couple_id in dev_couple_id_lst:
                # dev
                session_dev_list.append('hus.'+file)#+'_merged.npy')
                session_dev_list.append('wife.'+file)#+'_merged.npy')
            if couple_id in test_couple_id_lst:
                # test
                session_test_list.append('hus.'+file)#+'_merged.npy')
                session_test_list.append('wife.'+file)#+'_merged.npy')
        
        session_train_list_pair = [(os.path.join(self.feat_npy_dir, x+'.npy'), binary_score_dict[x]) for x in session_train_list if x in all_couple_rating_dict]
        session_dev_list_pair =[(os.path.join(self.feat_npy_dir, x+'.npy'), binary_score_dict[x]) for x in session_dev_list if x in all_couple_rating_dict]
        session_test_list_pair = [(os.path.join(self.feat_npy_dir, x+'.npy'), binary_score_dict[x]) for x in session_test_list if x in all_couple_rating_dict]
        
        return session_train_list_pair, session_dev_list_pair, session_test_list_pair
    
    def gen_test_list(self):
        return self.session_test_list_pair
    def gen_train_list(self):
        return self.session_train_list_pair
    def gen_dev_list(self):
        return self.session_dev_list_pair
    def gen_beh_list(self):
        return self.beh_list
    def return_dev_id_list(self):
        return self.dev_couple_id
    def return_unique_couple_id(self):
        return self.couple_id_set
