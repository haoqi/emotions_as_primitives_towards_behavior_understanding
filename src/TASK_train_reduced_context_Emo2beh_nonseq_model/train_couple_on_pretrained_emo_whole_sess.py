#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os, sys
import random

import numpy as np
import pdb
import argparse
import importlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

# change all classes to w_mask version
from Data_loader_couple_class_whole_sentence import Couple_datasest_torch_train_2_classes_whole_session
from Couple_data_class import Couple_info_class_binary_w_mask_split_dev

from tensorboardX import SummaryWriter

from utils import *


parser = argparse.ArgumentParser(description='PyTorch train for couple supervised')
# Env options:
parser.add_argument('--epochs', type=int, default=350, metavar='N',
                    help='number of iteration to train')
parser.add_argument('--expName', type=str, default='train_debug', metavar='E',
                    help='Experiment name')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')

# Model options
parser.add_argument('--topModelName', type=str, default='Couple_top_1d_NN_v1_input_full_session_binary_classification',
                    help='model class name')


parser.add_argument('--batchSize', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate')

parser.add_argument('--contiCount', type=str, default='0',
                    help='the number of continue time.')
parser.add_argument('--testCoupleID', type=str, default='105',
                    help='the id of test couple.')
parser.add_argument('--numDev', type=int, default=15,
                    help='number of dev couple id')
# Init args
torch.manual_seed(1234)
args = parser.parse_args()
args.expName = os.path.join('ckpts_couple_pre_emo_2classes_binary_sepDev_trainNo1', args.expName, args.testCoupleID)

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

if '-' in args.testCoupleID:
    fix_test_couple_id=args.testCoupleID.split('-')
else:
    fix_test_couple_id=[args.testCoupleID]
    
    
#NN_pretrained_emo_class_module = importlib.import_module('Model_class_pre_trained_emotion_cmu')
NN_top_couple_class_module = importlib.import_module('Model_class_top_couple')
    


def train_iter(model, optimizer, criterion, epoch, data_feat_torch, target_label_torch, mask, len_mask):
    '''
    # return
    # loss value, predect emotion torch
    The number of output of NN should be No.beh* 3
    '''
    
    threshold  = 0
    model.train()
    data_feat_torch = data_feat_torch.permute(0,2,1).to(device) # batch, channel, length
    target_label_torch = target_label_torch.to(device)
    mask = mask.to(device)

    optimizer.zero_grad()
    pred = model(data_feat_torch, len_mask) # batch* num_emo
    
    # find the number of non-zero elements
    total_num_binary_sample = torch.nonzero(mask.data>0).shape[0]
    
    loss = 0
    loss = criterion(pred, target_label_torch)
    loss = loss * mask
    loss = torch.sum(loss)/total_num_binary_sample
    
    loss.backward()
    optimizer.step()
    
    # check correct numbers    
    mask_select = mask.ge(0.1)    
    pred_binary_value  = torch.masked_select(pred, mask_select)
    target_label_torch_binary = torch.masked_select(target_label_torch, mask_select)
    
    pred_binary_label = pred_binary_value >= threshold
    correct_num_binary_sample = torch.sum(target_label_torch_binary.int() == pred_binary_label.int()).item()
    

    # calculate the classification accu
    return loss.item(), pred, total_num_binary_sample,  correct_num_binary_sample 

def eval_iter_per_session(model, data_feat_torch, mask, len_mask):
    #return list of predict label
    model.eval()
    #pdb.set_trace()
    data_feat_torch = data_feat_torch.permute(0,2,1).to(device)
    mask = mask.to(device)
    output_dev = model(data_feat_torch, len_mask)
    #value_dev, index_dev = torch.max(output_dev,dim=1)

    return output_dev.ge(0), output_dev

def eval_accu(mask, ref, pred):
    mask = mask.to(device)
    ref = ref.to(device)
    # find the number of non-zero elements
    total_num_binary_sample = torch.nonzero(mask.data>0).shape[0]
    
    # check correct numbers    
    mask_select = mask.ge(0.1)    # split 0 and 1 use 0.1 as threshold
    pred_binary_label  = torch.masked_select(pred, mask_select)
    target_label_torch_binary = torch.masked_select(ref, mask_select)
    
    #pred_binary_label = pred_binary_value >= 0
    correct_num_binary_sample = torch.sum(target_label_torch_binary.int() == pred_binary_label.int()).item()
    return total_num_binary_sample, correct_num_binary_sample, pred_binary_label


def numpy_append_0s(a, max_value):
    #pdb.set_trace()
    result = np.zeros((len(a), int(max_value), a[0].shape[1]))
    for i in range(len(a)):
        result[i , : a[i].shape[0], :] = a[i]
    return result.astype('float32')

def helper_collate(x, i, max_value):
    if i == 0 :
        return default_collate(numpy_append_0s(x, max_value))
    else:
        return default_collate(x)
    
def my_collate_1(batch):
    len_each_sample = [x[3] for x in batch]
    max_len = max(len_each_sample)
    transposed = zip(*batch)
    return [helper_collate(samples, i, max_len) for i, samples in enumerate(transposed)]
    

def main():
    logging.info("Prepare of couple data for training...")
    beh_list = ['acceptance', 'blame', 'positive', 'negative', 'sadness']
    
    couple_info_class = Couple_info_class_binary_w_mask_split_dev(given_test_list=fix_test_couple_id, num_dev_couple=NUM_DEV)
    dev_couple_id_list = couple_info_class.return_dev_id_list()
    interpolation_dict_str = couple_info_class.return_middle_split_dict()
    
    with open(os.path.join(args.expName, 'dev_couple_id.txt'), 'w') as f_out:
        f_out.write(' '.join(dev_couple_id_list) +'\n')
    print(dev_couple_id_list)

    dataset_train_torch = Couple_datasest_torch_train_2_classes_whole_session(couple_info_class.gen_train_list(), interpolation_dict_str,\
                                                 flag_load_all_feat_into_mem=True)
    dataset_dev_torch = Couple_datasest_torch_train_2_classes_whole_session(couple_info_class.gen_dev_list(), \
                                                interpolation_dict_str,  flag_load_all_feat_into_mem=True)
    dataset_test_torch = Couple_datasest_torch_train_2_classes_whole_session(couple_info_class.gen_test_list(), \
                                                interpolation_dict_str, flag_load_all_feat_into_mem=True)
    
    #pdb.set_trace()
    
    # define dataloader
    #-------------------------------
    dataloader_train = torch.utils.data.DataLoader(dataset_train_torch, \
                       batch_size=args.batchSize, collate_fn=my_collate_1,  shuffle=True, pin_memory=True)
    
    dataloader_dev = torch.utils.data.DataLoader(dataset_dev_torch, \
                      batch_size=args.batchSize,collate_fn=my_collate_1, shuffle=False, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test_torch, \
                       batch_size=1, shuffle=False,  pin_memory=True)
    
    
    
    #---------------------------------------------------
    # build the bottom NN with pretrained model
    #---------------------------------------------------
    # load pretrained cmu model

    #load the pretrained model
    pretrained_ckpt_test = '/auto/rcf-proj3/pg/haoqili/workspace_2018_icassp/src/TASK_train_1dCNN_cmu_kaldi_featureset/ckpts_cnn1d_cmumosei/try2_v2_adapt_lr_epoch1500_clear_grad/'+\
                    'epoch_1320--model.pth'
    pretrained_dict= torch.load(pretrained_ckpt_test)['state_dict']
    
    #---------------------------------------------------
    # build top models
    #---------------------------------------------------
    top_couple_model_name = getattr(NN_top_couple_class_module, args.topModelName)
    model = top_couple_model_name(input_feat_dim).to(device)
    
    #pdb.set_trace()
    curr_whole_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in curr_whole_model_dict}
    curr_whole_model_dict.update(pretrained_dict)
    model.load_state_dict(curr_whole_model_dict)
        
    # freeeze bottom layers
    for param in model.cnn_1d.parameters():
        param.requires_grad =  False
        
    init_lr = args.lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr)
    #weight_loss = torch.tensor([2*(2320)/(5340),2*(2320)/(5340),(5340-2320*2)/(5340)])
    criterion = nn.BCEWithLogitsLoss(reduction='none').to(device) 

    start_epoch = 0
    
    # Load ckpt if existsbinary_label_mask
    #--------------------------------------
    if args.checkpoint != '':
        logging.info('Load the ckpt form ' + args.checkpoint)
        #checkpoint_args_path = os.path.join(os.path.dirname(args.checkpoint) , 'args.pth')
        checkpoint_args = torch.load(args.checkpoint)
    
        start_epoch = checkpoint_args[1] # show at last
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        optimizer.load_state_dict(torch.load(args.checkpoint)['optimizer_dict'])
    
    #TRAINING
    best_epoch = 0
    for epoch in range(start_epoch,start_epoch + args.epochs):
        print ("epoch is {}".format(epoch))
        loss_train_epoch_total =0
        
        #reduce the learning rate
        curr_lr = poly_lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter=2, max_iter=args.epochs, power=0.9)
        
        num_iter = 0

        total_num_binary_classification_number = 0
        binary_correct_num_classification_number = 0
        # training for each iteration
        for iter_idx, (feat_in_torch, label_in_torch, binary_label_mask, len_mask, name) in enumerate(dataloader_train):
            #pdb.set_trace()
            train_loss_value_iter, pred_value, classification_sample_count, binary_classification_correct_count = \
                    train_iter(model, optimizer, criterion, epoch, feat_in_torch, label_in_torch, binary_label_mask, len_mask)
            
            if train_loss_value_iter !=0 :
                loss_train_epoch_total += train_loss_value_iter
                num_iter +=1

            total_num_binary_classification_number += classification_sample_count# count the number of binary labels 
            binary_correct_num_classification_number += binary_classification_correct_count
            
        
        if epoch % 5 == 0 :
            # training accu and loss
            logging.info('epoch '+str(epoch) + ' training_loss: ' + "{0:.3f}".format(loss_train_epoch_total/num_iter))
            logging.info('epoch '+str(epoch) + ' training_binary_classification_accu: ' + "{0:.3f}".format(binary_correct_num_classification_number/total_num_binary_classification_number))
            writer_board.add_scalar('training/loss', loss_train_epoch_total/num_iter, epoch)
            writer_board.add_scalar('training/binary_beh_accu', binary_correct_num_classification_number/total_num_binary_classification_number, epoch)
            
            #pdb.set_trace()
            # check dev loss
            #-------------------
            dev_accu_count_binary = 0
            dev_total_beh_task_binary = 0
            
            
            # calculate binary accu and 3 class accu
            for iter_dev, (feat_torch, label_torch, binary_label_mask, len_mask, name) in enumerate(dataloader_dev):
                # feat_torch : 1*1*len*dimension
                with torch.no_grad():
                    ouput_predict_label, dev_sample_pred_scores = eval_iter_per_session(model, feat_torch, binary_label_mask, len_mask)
                #print(all_sample_pred_scores)
                #print (name)
                #print(ouput_predict_label)
                #print('---')
                #print(label_torch)
                
                total_num_samples, correct_num_samples, pred_label_output = eval_accu(binary_label_mask, label_torch, ouput_predict_label)
                dev_accu_count_binary += correct_num_samples
                dev_total_beh_task_binary += total_num_samples

            logging.info('epoch '+str(epoch) + ' dev_binary_beh_classification: ' + '{0:.3f} numOfTest {1}'.format(dev_accu_count_binary/dev_total_beh_task_binary, dev_total_beh_task_binary))            
            writer_board.add_scalar('dev/binary_beh_accu', dev_accu_count_binary/dev_total_beh_task_binary, epoch)

            # save the model
            torch.save({'state_dict': model.state_dict(),\
                        'optimizer_dict': optimizer.state_dict()},\
                        '{}/epoch_{}--model.pth'.format(args.expName, str(epoch)))
            
            
            test_accu_count_binary = 0
            test_total_beh_task_binary = 0
            save_dict={}
            # test on test
            for iter_test, (feat_torch, label_torch, binary_label_mask, len_mask, name) in enumerate(dataloader_test):
                # feat_torch : 1*1*len*dimension
                test_ouput_predict_label, test_all_sample_pred_scores = eval_iter_per_session(model, feat_torch, binary_label_mask, len_mask)
                #print(all_sample_pred_scores)
                test_ouput_predict_beh_list = test_ouput_predict_label[0].tolist()
                ref_label = label_torch[0].tolist()
                save_dict[name[0].split('.npy')[0]] = test_ouput_predict_beh_list
                #pdb.set_trace()
                for i, ref_beh_score in enumerate(ref_label):
                    if ref_beh_score != 2 :
                        test_total_beh_task_binary+=1
                        if ref_beh_score == test_ouput_predict_beh_list[i]:
                            test_accu_count_binary +=1
            writer_board.add_scalar('test/binary_beh_accu', test_accu_count_binary/test_total_beh_task_binary, epoch)
            logging.info('epoch '+str(epoch) + ' test: binary_beh_classification: ' + '{0:.3f} numOfTest {1}'.format(test_accu_count_binary/test_total_beh_task_binary, test_total_beh_task_binary))    
            
            
            if dev_accu_count_binary/dev_total_beh_task_binary >= best_epoch:
                best_epoch = dev_accu_count_binary/dev_total_beh_task_binary
                
                torch.save({'state_dict': model.state_dict(),\
                        'optimizer_dict': optimizer.state_dict()},\
                        '{}/best_epoch_{}--model.pth'.format(args.expName, str(epoch)))
                torch.save({'state_dict': model.state_dict(),\
                        'optimizer_dict': optimizer.state_dict()},\
                        '{}/best_ckpt--model.pth'.format(args.expName, str(epoch)))
              
                logging.info('save the MODEL for test')            
                np.save('{}/test_best_result.npy'.format(args.expName), save_dict)

                
if __name__ =='__main__':
    main()
