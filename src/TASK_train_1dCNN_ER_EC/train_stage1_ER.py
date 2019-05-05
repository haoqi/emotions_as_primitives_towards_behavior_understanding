#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:13:22 2018

@author: haoqi
"""

import os
import random
import glob
import numpy as np
import pdb
import argparse
import importlib
import math

from cmu_mosei_class import CMU_MOSEI_dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
from utils import *

NN_class_module = importlib.import_module('model_cnn1d_cmu_stage1')

parser = argparse.ArgumentParser(description='PyTorch train on cmu mosei')

# Env options:
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of iteration to train')
parser.add_argument('--expName', type=str, default='train_debug', metavar='E',
                    help='Experiment name')
parser.add_argument('--modelName', type=str, default='Base_1D_NN_fixed_seq_len_1s_majvote',
                    help='CNN model')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--lr', type=float, default=0.000001,
                    help='Learning rate')

parser.add_argument('--contiCount', type=str, default='0',
                    help='the number of continue time.')
parser.add_argument('--minLength', type=float, default=1,
                    help='len of frame')

parser.add_argument('--batchSize', type=int, default=16,
                    help='batch s')

# Initialize args
#---------------------------
torch.manual_seed(1234)
args = parser.parse_args()

args.expName = os.path.join('ckpts_cnn1d_cmumosei', args.expName)
    
if args.contiCount != '0':    
    writer_board = SummaryWriter(os.path.join(args.expName, 'tensorboardX_continue_'+args.continueCount))
else:
    writer_board = SummaryWriter(os.path.join(args.expName, 'tensorboardX'))
    
logging = create_output_dir_w_logging(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_len_emo_audio = args.minLength

data_npy_root_dir = '../../data_cmu_mosei/trimmed_utterance_kaldi_featureset/'
input_feat_dim = 84
num2emodict = {0:'hap', 1:'sad', 2: 'anger', 3:'fear', 4:'disgust', 5:'surprise'} 

# IEMOCAP dataset class
#----------------------------
class CMU_Dataset_train_torch(Dataset):
    '''
    torch_Dataset for CMU dataset for training
    if utterance is shorter than sample_win_len append 0
    else randomly select one sample_win length sample form the whole utterance
    
    '''
    def __init__(self, CMU_dataset_info_class, sample_win):
        # length of training sample(s)
        self.sample_win_len = int(sample_win * 100) # 100 frames per second
        self.train_utterance_list = CMU_dataset_info_class.gen_train_pair() # [(Sess_utterance, label)...]
        
        # load all training data into memory
        self.training_feat_all_dict = {}
        for item in self.train_utterance_list:
            npy_feat = np.load(os.path.join(data_npy_root_dir, item[0]+'.npy')).astype('float32')
            self.training_feat_all_dict[item[0]] = npy_feat
    def __len__(self):
        return len(self.train_utterance_list)
    
    def __getitem__(self, idx):
        sess = self.train_utterance_list[idx][0]
        label_curr = self.train_utterance_list[idx][1]
        len_curr = self.training_feat_all_dict[sess].shape[0]
        #print(len_curr)
        
        if len_curr >= self.sample_win_len:
            start_idx = self.select_random_windows(len_curr, self.sample_win_len)
            return (self.training_feat_all_dict[sess][start_idx:start_idx+self.sample_win_len,:], 
                    label_curr)
        else:
            # the session is shorter than self.sample_win_len
            # append zeros at the end of that utterance
            feat_dim = self.training_feat_all_dict[sess].shape[1]
            return_feat = np.zeros((self.sample_win_len, feat_dim))
            return_feat[:len_curr, :feat_dim] = self.training_feat_all_dict[sess]
            
            return(return_feat.astype('float32'), label_curr)
    
    def select_random_windows(self, total_len, win_sz):
        # return the index
        return np.random.randint(0, total_len-win_sz+1)
   
class CMU_Dataset_dev_test_torch(Dataset):
    '''
    
    for dev and test data, evaluate the performance on each frame, and then do majority vote
    self.dev_utterance_list = IEMOCAP_dataset_info_class.gen_dev_pair() # [(Sess_utterance, label)...]
    self.test_utterance_list = IEMOCAP_dataset_info_class.gen_eval_pair()
    '''
    def __init__(self, input_pair_list, sample_win):
        self.sample_win_len = int(sample_win * 100) # 100 frames per second
        self.utterance_list = input_pair_list# [(Sess_utterance, label)...]
        # load all data into memory
        self.feat_all_dict = {}
        for item in self.utterance_list:
            npy_feat = np.load(os.path.join(data_npy_root_dir, item[0]+'.npy')).astype('float32')
            self.feat_all_dict[item[0]] = npy_feat
        
    def __len__(self):
        return len(self.utterance_list)
    def __getitem__(self, idx):
        
        sess = self.utterance_list[idx][0]
        label_curr = self.utterance_list[idx][1]
        
        len_curr = self.feat_all_dict[sess].shape[0]
        feat_dim = self.feat_all_dict[sess].shape[1]
        
        int_num_slices = len_curr / self.sample_win_len
        
        
        if len_curr < self.sample_win_len:
            total_slice_num =1
            return_feat = np.zeros((1, int(self.sample_win_len), int(feat_dim)))
            return_feat[0, 0:len_curr, :] = self.feat_all_dict[sess][0:len_curr,:]
                
        else:        
            if int_num_slices - math.floor(int_num_slices) < 0.3: # ignore the last part
                total_slice_num = math.floor(int_num_slices)
                flag_ignore = True
            else:
                total_slice_num = math.floor(int_num_slices)+1
                flag_ignore = False
            
            return_feat = np.zeros((int(total_slice_num), int(self.sample_win_len), int(feat_dim)))       
            for i in range(total_slice_num):
                if i == total_slice_num-1:
                    if flag_ignore:
                        return_feat[i, :, :] = self.feat_all_dict[sess][i*self.sample_win_len:(i+1)*self.sample_win_len,:]
                    else:
                        return_feat[i, 0:len_curr-i*self.sample_win_len, :] = self.feat_all_dict[sess][i*self.sample_win_len:len_curr,:]
                        
                else:
                    return_feat[i, :, :] = self.feat_all_dict[sess][i*self.sample_win_len:(i+1)*self.sample_win_len,:]
            
        return (return_feat.astype('float32'), label_curr, total_slice_num)
                

    
def train_iter(model, optimizer, criterion, epoch, data_feat_torch, target_label_torch):
    # return
    # loss value, predect emotion torch
    model.train()
    data_feat_torch = data_feat_torch.permute(0,2,1).to(device) # batch, channel, length
    target_label_torch = target_label_torch.to(device)

    optimizer.zero_grad()
    
    pred = model(data_feat_torch) # batch* num_emo
    loss = criterion(pred, target_label_torch)

    loss.backward()
    optimizer.step()
    
    # check accu of training 
    return loss.item(), pred# number, torch

def eval_iter_per_utterance(model, data_feat_torch):
    #return list of predict label
    model.eval()
    data_feat_torch = data_feat_torch.squeeze(0).permute(0,2,1).to(device)
    
    output_dev = model(data_feat_torch)
    #value_dev, index_dev = torch.max(output_dev,dim=1)
    mean_value = torch.mean(output_dev, dim=0)
    
    return mean_value

       
def main():
    
    logging.info("Prepare of data for training...")
    

    dataset_cmu_mosei= CMU_MOSEI_dataset()
        
    
    # define dataset class for dataloader
    dataset_train_torch = CMU_Dataset_train_torch(dataset_cmu_mosei, min_len_emo_audio)
    dataset_dev_torch = CMU_Dataset_dev_test_torch(dataset_cmu_mosei.gen_dev_pair(), min_len_emo_audio)
    dataset_test_torch = CMU_Dataset_dev_test_torch(dataset_cmu_mosei.gen_eval_pair(), min_len_emo_audio)
    
    # define dataLoader
    dataloader_cmu = torch.utils.data.DataLoader(dataset_train_torch, \
                        batch_size=args.batchSize, shuffle=True, num_workers=4, pin_memory=True)
    
    # since the test are using majority vote, so the batch_size must set it to be 1 in our case
    dataloader_cmu_dev = torch.utils.data.DataLoader(dataset_dev_torch, \
                        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    dataloader_cmu_test = torch.utils.data.DataLoader(dataset_test_torch, \
                        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    
    # Init model
    #--------------------------------------
    nn_model_name  = getattr(NN_class_module, args.modelName)
    model = nn_model_name(input_feat_dim).to(device)
    init_lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    criterion = nn.MSELoss().to(device)
    start_epoch = 0

    
    # Load ckpt if exists
    #--------------------------------------
    if args.checkpoint != '':
        logging.info('Load the ckpt form ' + args.checkpoint)
        #checkpoint_args_path = os.path.join(os.path.dirname(args.checkpoint) , 'args.pth')
        checkpoint_args = torch.load(args.checkpoint)

        start_epoch = checkpoint_args[1] # show at last
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        optimizer.load_state_dict(torch.load(args.checkpoint)['optimizer_dict'])
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print ("epoch is {}".format(epoch))
        loss_train_epoch_total =0
        
        #update learning rate
        curr_lr = poly_lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter=2, max_iter=args.epochs, power=0.9)
        #print ("lr is {}".format(curr_lr))
        # training for each iteration
        for iter_idx,(feat_in_torch, label_in_torch) in enumerate(dataloader_cmu):
            train_loss_value_iter, pred_value = train_iter(model, optimizer, criterion, epoch, feat_in_torch, label_in_torch)
            loss_train_epoch_total += train_loss_value_iter
        
        if epoch % 2 == 0:
            # training accu and loss
            num_iter = len(dataloader_cmu)
            logging.info('epoch '+str(epoch) + ' training_loss: ' + "{0:.2f}".format(loss_train_epoch_total/num_iter))
            writer_board.add_scalar('training/loss', loss_train_epoch_total/num_iter, epoch)

            # check dev loss
            #-------------------
            dev_loss = 0
            for iter_dev, (feat_torch, label_torch, length_torch) in enumerate(dataloader_cmu_dev):
                # feat_torch : 1*1*len*dimension
                pred_score_mean = eval_iter_per_utterance(model, feat_torch)

                # calcualte the mse for each emotion
                dev_loss += (label_torch[0]-pred_score_mean.cpu())**2
                    
            num_file_dev = dataset_dev_torch.__len__()
            for i in range(6):
                writer_board.add_scalar('dev/'+num2emodict[i], dev_loss[i]/num_file_dev, epoch)
            

           # check test loss
            #-------------------
            test_loss = 0
            for iter_test, (feat_torch, label_torch, length_torch) in enumerate(dataloader_cmu_test):
                # feat_torch : 1*1*len*dimension
                pred_score_mean = eval_iter_per_utterance(model, feat_torch)
                #print(pred_score_mean)

                # calcualte the mse for each emotion
                test_loss += (label_torch[0]-pred_score_mean.cpu())**2
                    
            num_file_test= dataset_test_torch.__len__()
            
            for i in range(6):
                writer_board.add_scalar('test/'+num2emodict[i], test_loss[i]/num_file_test, epoch)
            
            # save the model
            torch.save({'state_dict': model.state_dict(),\
                        'optimizer_dict': optimizer.state_dict()},\
                        '{}/epoch_{}--model.pth'.format(args.expName, str(epoch))
		            
            # save the best model
            if best_dev_accu <= dev_whole_accu/num_file_dev:
                best_dev_accu = dev_whole_accu/num_file_dev
                logging.info('save best at epoch ' + str(epoch))
                # remove pre saved best ckpt
                os.system('rm -rf '+'{}/epoch_*_best.pth'.format(args.expName))
                # save new one
                torch.save({'state_dict': model.state_dict(),\
                    'optimizer_dict': optimizer.state_dict()}, '{}/epoch_{}_best.pth'.format(args.expName, epoch))
            
   
if __name__ =='__main__':
    main()     
