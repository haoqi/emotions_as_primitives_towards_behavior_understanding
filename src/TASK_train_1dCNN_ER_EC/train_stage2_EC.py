#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 14 20:05:31 2018

train on cmu mosei dataset base on first stage regression pre-training

This script is the training script of the second stage training 
for each emotion fixed the bottom layer from training stage 1, train top NN as 
a binary classification task for each emotion

Add args.balanceData to make training data balanced

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

from cmu_mosei_class import CMU_MOSEI_dataset_classification, CMU_MOSEI_dataset_classification_upsampling

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
from utils import *


NN_class_module = importlib.import_module('model_cnn1d_cmu_classification_stage2')

parser = argparse.ArgumentParser(description='PyTorch train on cmu mosei')
# Env options:
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of iteration to train')
parser.add_argument('--expName', type=str, default='train_debug', metavar='E',
                    help='Experiment name')
parser.add_argument('--modelName', type=str, default='Classification_Base_1D_NN_fixed_seq_len_1s_majvote_v2',
                    help='CNN model')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Learning rate')

parser.add_argument('--contiCount', type=str, default='0',
                    help='the number of continue time.')
parser.add_argument('--minLength', type=float, default=1,
                    help='len of frame')
parser.add_argument('--batchSize', type=int, default=32,
                    help='batch s')
parser.add_argument('--patience', type=int, default=15,
                    help='patience of stoping training')
parser.add_argument('--balanceData', type=str2bool, nargs='?', const=True,
                    default='False', help="whether or not upsampling under-represented class")
parser.add_argument('--weightedLoss', type=str2bool, nargs='?', const=True,
                    default='False', help="whether or not upsampling under-represented class")


# Initialize args
#---------------------------
torch.manual_seed(1234)
args = parser.parse_args()
args.expName = os.path.join('ckpts_cnn1d_cmumosei_classification', args.expName)
    
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
patience = args.patience
Flag_balanced_training_data = args.balanceData
Flag_weighted_loss = args.weightedLoss
all_emos_cmu = ['hap', 'sad', 'anger', 'fear', 'disgust', 'surprise']


def weighted_accuracy(TP, TN, P, N):
    # the function is used to compute weighted accuracy for classification of cmu mosei
    # http://aclweb.org/anthology/P17-1142
    return (TP*N/P + TN)/(2*N)

# torch based dataset class
#----------------------------
class CMU_Dataset_train_torch_classification(Dataset):
    '''
    torch_Dataset for CMU dataset for training
    if utterance is shorter than sample_win_len append 0
    else randomly select one sample_win length sample form the whole utterance
    
    '''
    def __init__(self, input_pair_list, sample_win):
        # length of training sample(s)
        self.sample_win_len = int(sample_win * 100) # 100 frames per second
        self.train_utterance_list = input_pair_list # [(Sess_utterance, label)...]
        
        # load all training data into memory
        self.training_feat_all_dict = {}
        for item in self.train_utterance_list:
            npy_feat = np.load(os.path.join(data_npy_root_dir, item[0]+'.npy')).astype('float32')
            self.training_feat_all_dict[item[0]] = npy_feat
            
    def __len__(self):
        return len(self.train_utterance_list)
    
    def select_random_windows(self, total_len, win_sz):
        # return the index
        return np.random.randint(0, total_len-win_sz+1)
    def __getitem__(self, idx):
        sess = self.train_utterance_list[idx][0]
        label_curr = self.train_utterance_list[idx][1]
        len_curr = self.training_feat_all_dict[sess].shape[0]
        
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
   
class CMU_Dataset_dev_test_torch_classification(Dataset):
    '''
    for dev and test data, evaluate the performance on each frame, and then do majority vote
    self.dev_utterance_list = CMU_MOSEI_dataset_classification.gen_dev_pair() # [(Sess_utterance, label)...]
    self.test_utterance_list = CMU_MOSEI_dataset_classification.gen_eval_pair()
    
    For this test data class, when using dataloaser, the batch_size can ONLY be set to 1
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
            # if the current utterance length is smaller than a single win len
            # set a whole zeros feature numpy and fill with the true non-zero feature values
            total_slice_num =1
            return_feat = np.zeros((1, int(self.sample_win_len), int(feat_dim)))
            return_feat[0, 0:len_curr, :] = self.feat_all_dict[sess][0:len_curr,:]          
        else:
            # the utterance is longer than a single win length
            
            if int_num_slices - math.floor(int_num_slices) < 0.3: # ignore the last part, since it is too short
                total_slice_num = math.floor(int_num_slices)
                flag_ignore = True
            else:
                # will append 0s for last part
                total_slice_num = math.floor(int_num_slices)+1
                flag_ignore = False
            
            # assign all 0s for the returned npy file
            return_feat = np.zeros((int(total_slice_num), int(self.sample_win_len), int(feat_dim)))       
            
            for i in range(total_slice_num):
                if i == total_slice_num-1: # last segment
                    if flag_ignore:
                        return_feat[i, :, :] = self.feat_all_dict[sess][i*self.sample_win_len:(i+1)*self.sample_win_len,:]
                    else:
                        return_feat[i, 0:len_curr-i*self.sample_win_len, :] = self.feat_all_dict[sess][i*self.sample_win_len:len_curr,:]                   
                else: # not the last segment
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
    
    value, index = torch.max(pred,dim=1)
    
    # check accu of training 
    return loss.item(), index# number, torch

def eval_iter_per_utterance(model, data_feat_torch):
    #return list of predict label
    model.eval()
    data_feat_torch = data_feat_torch.squeeze(0).permute(0,2,1).to(device)
    
    output_dev = model(data_feat_torch)
    value_dev, index_dev = torch.max(output_dev,dim=1)
    #mean_value = torch.mean(output_dev, dim=0)
    
    return index_dev.tolist()


def main():
    for emo_idx, emo_cmu in enumerate(all_emos_cmu):
        
        logging.info("Prepare of data for second stage training...")
        
        if Flag_balanced_training_data:
            # upsampling
            dataset_cmu_mosei= CMU_MOSEI_dataset_classification_upsampling()
            # define dataset class for dataloader
            dataset_train_torch = \
            CMU_Dataset_train_torch_classification(dataset_cmu_mosei.gen_balanced_train_pair(emo_idx), min_len_emo_audio)
        else:
                
            dataset_cmu_mosei= CMU_MOSEI_dataset_classification()
            # define dataset class for dataloader
            dataset_train_torch = CMU_Dataset_train_torch_classification(dataset_cmu_mosei.gen_train_pair(), min_len_emo_audio)
        dataset_dev_torch = CMU_Dataset_dev_test_torch_classification(dataset_cmu_mosei.gen_dev_pair(), min_len_emo_audio)
        dataset_test_torch = CMU_Dataset_dev_test_torch_classification(dataset_cmu_mosei.gen_eval_pair(), min_len_emo_audio)
        
        # define dataLoader
        dataloader_cmu = torch.utils.data.DataLoader(dataset_train_torch, \
                            batch_size=args.batchSize, shuffle=True, num_workers=6, pin_memory=True)
        
        # since the test are using majority vote, so the batch_size must set it to be 1 in our case
        dataloader_cmu_dev = torch.utils.data.DataLoader(dataset_dev_torch, \
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        dataloader_cmu_test = torch.utils.data.DataLoader(dataset_test_torch, \
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        
        
        logging.info('Training for emotion {}'.format(emo_cmu))

        # Init model
        #--------------------------------------
        nn_model_name  = getattr(NN_class_module, args.modelName)
        model = nn_model_name(input_feat_dim).to(device)
        init_lr = args.lr
        
        
        # Due to the class imblance, add weight
        #-------------------
        if Flag_weighted_loss:
            imbalance_count0, imbalance_count1 = dataset_cmu_mosei.count_imbalance_distribution(emo_idx)
            weight_CrossEntropyLoss = torch.tensor([imbalance_count1/(imbalance_count0+imbalance_count1), \
                                                    imbalance_count0/(imbalance_count0+imbalance_count1)])
            criterion = nn.CrossEntropyLoss(weight=weight_CrossEntropyLoss).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
            
        start_epoch = 0
        best_dev_accu = -1
        count_patience = 0
        
        # load the pretrain_model
        ckpt_test = '/mnt/ssd1/haoqi_ssd1/workspace_2018_ICASSP/src/TASK_train_1dCNN_cmu_kaldi_featureset/ckpts_cnn1d_cmumosei/try2_v2_adapt_lr_epoch1500_clear_grad/'+\
                    'epoch_1320--model.pth'
        
        pretrained_dict= torch.load(ckpt_test)['state_dict']
        curr_model_dict= model.state_dict()
        '''
        for param in model.parameters():
            print(param.shape)
            print(param.requires_grad)
        '''
        # fill out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in curr_model_dict}
        curr_model_dict.update(pretrained_dict)
        model.load_state_dict(curr_model_dict)
    
        # freeze pre-trained layers
        for param in model.parameters():
            #print(param.shape)
            param.requires_grad = False
        # optimize the last several layers
        for param in model.out2.parameters():
            param.requires_grad = True
        
        # check if those corresponding layers are freezed
        #for param in model.parameters():
        #    print(param.shape)
        #    print(param.requires_grad)        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr)
        
        #------------------------------------------
        # training the model
        #------------------------------------------
        for epoch in range(start_epoch, start_epoch + args.epochs):
            print ("epoch is {}".format(epoch))
            loss_train_epoch_total =0
            accu_train_classification =0
            
            #update learning rate
            curr_lr = poly_lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter=2, max_iter=args.epochs, power=0.9)
            #print ("lr is {}".format(curr_lr))
            # training for each iteration
            for iter_idx, (feat_in_torch, label_in_torch) in enumerate(dataloader_cmu):
                
                used_label_in_torch = label_in_torch[:, emo_idx]
                train_loss_value_iter, pred_value = \
                train_iter(model, optimizer, criterion, epoch, feat_in_torch, used_label_in_torch)
                loss_train_epoch_total += train_loss_value_iter
                #pdb.set_trace()
                accu_train_classification += torch.sum(used_label_in_torch==pred_value.cpu()).item()
                
            
            # save the model at the end of epoch
            if epoch % 2 == 0:
                # training accu and loss
                num_iter = len(dataloader_cmu)
                logging.info('epoch '+str(epoch) + ' training_loss: ' + "{0:.2f}".format(loss_train_epoch_total/num_iter))
                logging.info('epoch '+str(epoch) + ' training_accu: ' + "{0:.2f}".format(accu_train_classification/dataset_train_torch.__len__()))
                writer_board.add_scalar('training/'+emo_cmu+'_loss', loss_train_epoch_total/num_iter, epoch)
                writer_board.add_scalar('training/'+emo_cmu+'_classfication_accu', accu_train_classification/dataset_train_torch.__len__(), epoch)
                
                # check dev loss
                #-------------------
                dev_whole_accu = 0
                dev_weighted_accu = {'0':{'total':0, 'correct':0}, '1':{'total':0, 'correct':0}}
                dev_tgt_label_lst = []
                dev_test_label_lst = []
                
                for iter_dev, (feat_torch, label_torch, length_torch) in enumerate(dataloader_cmu_dev):
                    used_dev_label_in_torch = label_torch[:, emo_idx]
                    
                    # feat_torch : 1*1*len*dimension
                    pre_label_lst = eval_iter_per_utterance(model, feat_torch)
    
                    # do majority vote
                    pred_label_session = most_common(pre_label_lst)
                    dev_weighted_accu[str(used_dev_label_in_torch.cpu().item())]['total']+=1

                    if pred_label_session == used_dev_label_in_torch.cpu().item():
                        dev_whole_accu+=1
                        dev_weighted_accu[str(used_dev_label_in_torch.cpu().item())]['correct']+=1
                    
                    dev_tgt_label_lst.append(used_dev_label_in_torch.cpu().item())
                    dev_test_label_lst.append(pred_label_session)
                        
                #pdb.set_trace()        
                num_file_dev = dataset_dev_torch.__len__()
                logging.info('epoch '+str(epoch) + ' dev_accu: ' + "{0:.3f}".format(dev_whole_accu/num_file_dev))
                writer_board.add_scalar('dev/'+emo_cmu+'_accu', dev_whole_accu/num_file_dev, epoch)

                # calculate the f1 and weighted accu based on cmu-mosei paper
                dev_weighted_accuracy_value = weighted_accuracy(dev_weighted_accu['1']['correct'], dev_weighted_accu['0']['correct'], \
                                                      dev_weighted_accu['1']['total'], dev_weighted_accu['0']['total'])
                
                logging.info('epoch '+str(epoch) + ' weighted_dev_accu: ' + "{0:.3f}".format(dev_weighted_accuracy_value))
                # F1 score
                #pdb.set_trace()
                dev_f1_score_value = f1_score(dev_tgt_label_lst, dev_test_label_lst, average='weighted')
                logging.info('epoch '+str(epoch) + ' dev F1 score2: ' + "{0:.3f}".format(dev_f1_score_value))
                del dev_weighted_accu
                
                
                # check test loss
                #-------------------
                test_whole_accu = 0
                test_weighted_accu = {'0':{'total':0, 'correct':0}, '1':{'total':0, 'correct':0}}
                tgt_label_lst = []
                test_label_lst = []
                
                for iter_test, (feat_torch, label_torch, length_torch) in enumerate(dataloader_cmu_test):
                    used_test_label_in_torch = label_torch[:, emo_idx]
                    # feat_torch : 1*1*len*dimension
                    pre_label_lst = eval_iter_per_utterance(model, feat_torch)
    
                    # do majority vote
                    pred_label_session = most_common(pre_label_lst)
                    test_weighted_accu[str(used_test_label_in_torch.cpu().item())]['total']+=1
                    if pred_label_session == used_test_label_in_torch.cpu().item():
                        test_whole_accu+=1
                        test_weighted_accu[str(used_test_label_in_torch.cpu().item())]['correct']+=1
                    
                    tgt_label_lst.append(used_test_label_in_torch.cpu().item())
                    test_label_lst.append(pred_label_session)
                        
                num_file_test = dataset_test_torch.__len__()
                logging.info('epoch '+str(epoch) + ' test_accu: ' + "{0:.3f}".format(test_whole_accu/num_file_test))
                writer_board.add_scalar('test/'+emo_cmu+'_accu', test_whole_accu/num_file_test, epoch)
                #print(test_weighted_accu)
                # calculate the f1 and weighted accu based on cmu-mosei paper
                test_weighted_accuracy_value = weighted_accuracy(test_weighted_accu['1']['correct'], test_weighted_accu['0']['correct'], \
                                                      test_weighted_accu['1']['total'], test_weighted_accu['0']['total'])
                
                logging.info('epoch '+str(epoch) + ' weighted_test_accu: ' + "{0:.3f}".format(test_weighted_accuracy_value))
                
                
                # F1 score
                test_f1_score_value = f1_score(tgt_label_lst, test_label_lst, average='weighted')
                logging.info('epoch '+str(epoch) + ' test F1 score2: ' + "{0:.3f}".format(test_f1_score_value))
                del test_weighted_accu
                
                # save the model
                torch.save({'state_dict': model.state_dict(),\
                            'optimizer_dict': optimizer.state_dict()},\
                            '{}/{}_epoch_{}--model.pth'.format(args.expName, emo_cmu, str(epoch)))
                
                
                if dev_weighted_accuracy_value > best_dev_accu:
                    # save the best model 

                    # save the best model
                    logging.info('save best at epoch ' + str(epoch))
                    # remove pre saved best ckpt
                    os.system('rm -rf '+'{}/{}_epoch_*_best*.pth'.format(args.expName, emo_cmu))
                    # save new one
                    torch.save({'state_dict': model.state_dict(),\
                            'optimizer_dict': optimizer.state_dict()}, \
                            '{0}/{1}_epoch_{2}_best_accu{3:.3f}_f1_{4:.3f}.pth'.format(args.expName,emo_cmu,epoch,\
                            test_weighted_accuracy_value, test_f1_score_value))
                    
                    best_dev_accu = dev_weighted_accuracy_value
                    # reset patience count
                    count_patience=0
                count_patience +=1
                
                print('----{}'.format(count_patience))
                if count_patience> patience:
                    # stop training
                    break


if __name__ =='__main__':
    main()     

