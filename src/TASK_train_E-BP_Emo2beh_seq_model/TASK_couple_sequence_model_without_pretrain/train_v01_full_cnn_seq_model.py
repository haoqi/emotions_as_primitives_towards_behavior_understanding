#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:12:09 2019

@author: haoqi
"""


import os, sys


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import numpy as np
import pdb
import argparse
import importlib

# Couple_info_class_binary_w_mask_split_dev is a class to split couple training and dev based on given test
from Couple_data_class_seq import Couple_info_class_binary_w_mask_split_dev
from Dataloader_couple_cnn_sequence import Dataloader_couple_cnn_seq


from tensorboardX import SummaryWriter

from utils import *


parser = argparse.ArgumentParser(description='PyTorch train for couple supervised')
# Env options:
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of iteration to train')
parser.add_argument('--expName', type=str, default='train_debug', metavar='E',
                    help='Experiment name')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')

# Model options
parser.add_argument('--ModelName', type=str, default='model_cnn_seq_fully',
                    help='model class name')
#parser.add_argument('--baseModelName', type=str, default='pre_train_CMU_bottom_N2',
#                    help='model class name')

parser.add_argument('--batchSize', type=int, default=1,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Learning rate')

parser.add_argument('--contiCount', type=str, default='0',
                    help='the number of continue time.')
parser.add_argument('--testCoupleID', type=str, default='302-633-627-581',
                    help='the id of test couple.')
parser.add_argument('--numDev', type=int, default=10,
                    help='number of dev couple id')
# Init args
torch.manual_seed(1234)
args = parser.parse_args()
args.expName = os.path.join('ckpts_couple_seq_binary_trainNo1', args.expName, args.testCoupleID)

if args.contiCount != '0':    
    writer_board = SummaryWriter(os.path.join(args.expName, 'tensorboardX_continue_'+args.continueCount))
else:
    writer_board = SummaryWriter(os.path.join(args.expName, 'tensorboardX'))

logging = create_output_dir_w_logging(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_feat_dim = 84
NUM_DEV = args.numDev
beh_dict = {0:'acceptance', 1:'blame',2: 'positive', 3:'negative', 4:'sadness'}
#fix_test_couple_id = ['563', '314', '521', '302']
MAX_EARLY_STOP= 10
if '-' in args.testCoupleID:
    fix_test_couple_id=args.testCoupleID.split('-')
else:
    fix_test_couple_id=[args.testCoupleID]
    
NN_top_couple_class_module = importlib.import_module('Model_class_sequence_couple')

def train_iter(model, optimizer, criterion, epoch, data_feat_torch, target_label_torch, mask, total_non_masked_samples, threshold):

    model.train()
    #original (N, length, 84) --> (N, C_in, L_in)
    data_feat_torch = data_feat_torch.permute(0,2,1).to(device)
    target_label_torch = target_label_torch.to(device)
    mask = mask.to(device)
    
    optimizer.zero_grad()
    pred = model(data_feat_torch)
    
    loss = 0
    loss = criterion(pred, target_label_torch)
    loss = loss * mask
    loss = torch.sum(loss)/total_non_masked_samples
    
    loss.backward()
    optimizer.step()
    
    # eval the training process
    # ------------------------------------------
    mask_select = mask.ge(0.1)    
    pred_binary_value  = torch.masked_select(pred, mask_select)
    target_label_torch_binary = torch.masked_select(target_label_torch, mask_select)
    
    pred_binary_label = pred_binary_value >= threshold
    correct_num_binary_sample = torch.sum(target_label_torch_binary.int() == pred_binary_label.int()).item()
    
    return loss.item(), pred, correct_num_binary_sample


def eval_iter(model, criterion, epoch, data_feat_torch, target_label_torch, mask, total_non_masked_samples, threshold):
    model.eval()
    data_feat_torch = data_feat_torch.permute(0,2,1).to(device)
    target_label_torch = target_label_torch.to(device)
    mask = mask.to(device)
    
    pred = model(data_feat_torch)
    
    loss = 0
    loss = criterion(pred, target_label_torch)
    loss = loss * mask
    sum_loss = torch.sum(loss)
    
    # eval the training process
    # ------------------------------------------
    mask_select = mask.ge(0.1)    
    pred_binary_value  = torch.masked_select(pred, mask_select)
    target_label_torch_binary = torch.masked_select(target_label_torch, mask_select)
    
    pred_binary_label = pred_binary_value >= threshold
    correct_num_binary_sample = torch.sum(target_label_torch_binary.int() == pred_binary_label.int()).item()
    
    return sum_loss.item(), pred, correct_num_binary_sample

def eval_test(model, criterion, epoch, data_feat_torch, target_label_torch, mask, total_non_masked_samples, threshold):
    model.eval()
    data_feat_torch = data_feat_torch.permute(0,2,1).to(device)
    pred = model(data_feat_torch)
    return pred.ge(0), pred

def main():
    logging.info("Prepare of couple data for training...")
    beh_list = ['acceptance', 'blame', 'positive', 'negative', 'sadness']
    
    couple_info_class = Couple_info_class_binary_w_mask_split_dev(given_test_list=fix_test_couple_id, num_dev_couple=NUM_DEV)
    dev_couple_id_list = couple_info_class.return_dev_id_list()
    
    with open(os.path.join(args.expName, 'dev_couple_id.txt'), 'w') as f_out:
        f_out.write(' '.join(dev_couple_id_list) +'\n')
    print(dev_couple_id_list)

    dataset_train_torch = Dataloader_couple_cnn_seq(couple_info_class.gen_train_list())
    dataset_dev_torch = Dataloader_couple_cnn_seq(couple_info_class.gen_dev_list())
    dataset_test_torch = Dataloader_couple_cnn_seq(couple_info_class.gen_test_list())
    
    # define dataloader
    #-------------------------------
    dataloader_train = torch.utils.data.DataLoader(dataset_train_torch, \
                       batch_size=args.batchSize, shuffle=True, pin_memory=True)
    dataloader_dev = torch.utils.data.DataLoader(dataset_dev_torch, \
                      batch_size=args.batchSize, shuffle=False, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test_torch, \
                       batch_size=1, shuffle=False,  pin_memory=True)
    

    #build the models
    #-------------------------------
    selected_model_name = getattr(NN_top_couple_class_module, args.ModelName)
    model = selected_model_name(input_feat_dim).to(device)

    init_lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    ## the loss will not average for 
    #------------------------------
    criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
    eval_threshold = 0 # the eval threshold ==0 if using BCEWithLogitsLoss

    # TRAINING stage
    # not from pretrained ckpt
    best_epoch_dev_accu = 0
    start_epoch = 0
    
    early_stop = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print("epoch is {}".format(epoch))
        
        if early_stop > MAX_EARLY_STOP:
            break
        
        #reduce the learning rate
        curr_lr = poly_lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter=2, max_iter=args.epochs+10, power=0.9)
        
        total_num_binary_classification_behs = 0
        total_binary_correct_num_classification_number = 0
        loss_train_epoch_total = 0
        
        for iter_idx, (feat_in_torch, label_in_torch, binary_label_mask, len_mask, name) in enumerate(dataloader_train):
            # feat_in_torch: shape[batch=1, length, win_sz =100*1, feat_dim84]
            # label_in_torch: shape[batch, 5] 5 behaviors in total
            # binary_label_mask: shape[batch, 5] 5 behaviors in total
            #pdb.set_trace()
            #remove the first dimension caused by data_loader
            feat_in_torch = torch.squeeze(feat_in_torch, dim=0)
            
            total_num_binary_samples_current_batch = torch.nonzero(binary_label_mask.data>0).shape[0]
            if total_num_binary_samples_current_batch ==0:
                continue
            else:
                train_loss_value_iter, pred_value, binary_classification_correct_count =\
                    train_iter(model, optimizer, criterion, epoch, feat_in_torch, label_in_torch, binary_label_mask, total_num_binary_samples_current_batch,eval_threshold)
                #pdb.set_trace()
                total_binary_correct_num_classification_number += binary_classification_correct_count
                total_num_binary_classification_behs += total_num_binary_samples_current_batch
                
                loss_train_epoch_total += train_loss_value_iter*total_num_binary_samples_current_batch
        if epoch % 5 == 0:
            # training accu and loss
            logging.info('epoch '+str(epoch) + ' training_loss: ' + "{0:.3f}".format(loss_train_epoch_total/total_num_binary_classification_behs))
            logging.info('epoch '+str(epoch) + ' training_binary_classification_accu: ' + "{0:.3f}".format(total_binary_correct_num_classification_number/total_num_binary_classification_behs))
            writer_board.add_scalar('training/loss', loss_train_epoch_total/total_num_binary_classification_behs, epoch)
            writer_board.add_scalar('training/binary_beh_accu', total_binary_correct_num_classification_number/total_num_binary_classification_behs, epoch)
            
            # Evaluate on dev set
            dev_loss_total = 0
            dev_accu_count_binary = 0
            dev_total_beh_task_binary = 0
            for iter_idx_dev, (dev_feat_in_torch, dev_label_in_torch, dev_binary_label_mask, dev_len_mask, dev_name) in enumerate(dataloader_dev):
                dev_total_num_binary_samples_current_batch = torch.nonzero(dev_binary_label_mask.data>0).shape[0]
                #remove the first dimension caused by data_loader
                dev_feat_in_torch = torch.squeeze(dev_feat_in_torch, dim=0)
                
                if dev_total_num_binary_samples_current_batch == 0:
                    continue
                else:
                    dev_sum_loss, dev_pred_value, dev_binary_classification_correct_count = \
                        eval_iter(model, criterion, epoch, dev_feat_in_torch, dev_label_in_torch, dev_binary_label_mask, dev_total_num_binary_samples_current_batch, eval_threshold)
                    dev_loss_total += dev_sum_loss
                    dev_accu_count_binary += dev_binary_classification_correct_count
                    dev_total_beh_task_binary+= dev_total_num_binary_samples_current_batch
            
            logging.info('epoch '+str(epoch) + ' dev_loss: ' + "{0:.3f}".format(dev_loss_total/dev_total_beh_task_binary))
            logging.info('epoch '+str(epoch) + ' dev_binary_beh_classification: ' + '{0:.3f} numOfTest {1}'.format(dev_accu_count_binary/dev_total_beh_task_binary, dev_total_beh_task_binary))            
            writer_board.add_scalar('dev/binary_beh_accu', dev_accu_count_binary/dev_total_beh_task_binary, epoch)

            # save the model
            torch.save({'state_dict': model.state_dict(),\
                        'optimizer_dict': optimizer.state_dict()},\
                        '{}/epoch_{}--model.pth'.format(args.expName, str(epoch)))
            
            # test on test data
            test_accu_count_binary = 0
            test_total_beh_task_binary = 0
            save_dict={}
            
            
            for iter_idx_test, (test_feat_in_torch, test_label_in_torch, test_binary_label_mask, test_len_mask, test_name) in enumerate(dataloader_test):
                test_total_num_binary_samples_current_batch = torch.nonzero(test_binary_label_mask.data>0).shape[0]
                #remove the first dimension caused by data_loader
                test_feat_in_torch = torch.squeeze(test_feat_in_torch, dim=0)
                if test_total_num_binary_samples_current_batch == 0:
                    continue
                else:
                    test_ouput_predict_label, test_all_sample_pred_scores = \
                        eval_test(model, criterion, epoch, test_feat_in_torch, test_label_in_torch, test_binary_label_mask, test_total_num_binary_samples_current_batch, eval_threshold)
                
                test_ouput_predict_beh_list = test_ouput_predict_label[0].tolist()
                ref_label = test_label_in_torch[0].tolist()
                save_dict[test_name[0]] = test_ouput_predict_beh_list
                #pdb.set_trace()
                for i, ref_beh_score in enumerate(ref_label):
                    if ref_beh_score != 2 :
                        test_total_beh_task_binary+=1
                        if ref_beh_score == test_ouput_predict_beh_list[i]:
                            test_accu_count_binary +=1
                    
            writer_board.add_scalar('test/binary_beh_accu', test_accu_count_binary/test_total_beh_task_binary, epoch)
            logging.info('epoch '+str(epoch) + ' test: binary_beh_classification: ' + '{0:.3f} numOfTest {1}'.format(test_accu_count_binary/test_total_beh_task_binary, test_total_beh_task_binary))    
            
            
            # save the best model based on the performance of dev set
            if dev_accu_count_binary/dev_total_beh_task_binary > best_epoch_dev_accu:
                early_stop = 0 
                best_epoch_dev_accu = dev_accu_count_binary/dev_total_beh_task_binary
                
                torch.save({'state_dict': model.state_dict(),\
                        'optimizer_dict': optimizer.state_dict()},\
                        '{}/best_epoch_{}--model.pth'.format(args.expName, str(epoch)))
                torch.save({'state_dict': model.state_dict(),\
                        'optimizer_dict': optimizer.state_dict()},\
                        '{}/best_ckpt--model.pth'.format(args.expName, str(epoch)))
              
                logging.info('save the MODEL for test')            
                np.save('{}/test_best_result.npy'.format(args.expName), save_dict)
            else:
                early_stop += 1
                print('early_stop' + str(early_stop))

if __name__ == '__main__':

    main()


