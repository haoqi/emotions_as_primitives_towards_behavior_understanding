#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:57:52 2018

This function is to merge feature extracted from kaldi

the feature set including energy, mfb, mfcc, pitch 
feat_set: 0, 1, 2 pitch
3, energy
4-43 mfb
44-83 mfcc
@author: haoqi
"""
import sys
import os
import glob

from kaldi_io import *

import subprocess
import tempfile
import logging
import shlex
import numpy as np
import argparse
import json
import pdb


input_wav_files = glob.glob('./output_debug/*.wav')
ouput_dir_path = './output_debug'
os.system('mkdir -p '+ouput_dir_path)
SAVE_FORMAT = 'np'


# Select kaldi,
if not 'KALDI_ROOT' in os.environ:
  # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
  os.environ['KALDI_ROOT']='/mnt/ssd2/haoqi_ssd/workspace_2018_fall/software/kaldi'

# Add kaldi tools to path,
os.environ['PATH'] = os.popen('echo $KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/').readline().strip() + ':' + os.environ['PATH']


def tryremove(path):
    try:
        os.remove(path)
    except:
        pass


def run(files, output_path, batch_size=50, normalize=True, num_mfcc=13,
        num_mfb=23, energy=True):
    # split wav file list into filelist_batch
    global SAVE_FORMAT 
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    for files_batch in batches:
        new_res = extract_features(files_batch, normalize=normalize, num_mfcc=num_mfcc,
                                   num_mfb=num_mfb, energy=energy)
        if SAVE_FORMAT == 'np':
            for f in new_res:
                np.save(os.path.join(output_path,f), new_res[f])


def extract_features(files, normalize=True,
                     delta=0, spk2utt=None, num_mfcc=13, num_mfb=23, energy=True):
    """Extract speech features using kaldi
    Parameters:
    files: list
    normalize: bool, do mean variance normalization
    delta: int, [0..2], 0 -> no deltas, 1 -> do deltas, 2 -> do delta+deltasdeltas
    pitch: bool, compute pitch
    energy: do calculate energy when compute mfb
    """
    
    #TODO add delta params
    try:
        # writing 'file path' in a .scp file for kaldi
        def get_fname(path):
            return os.path.basename(path).split('.wav')[0]
        # create file scp
        (scpid, scpfn) = tempfile.mkstemp()
        with open(scpfn, 'w') as fout:
            fout.write('\n'.join((' '.join([get_fname(f), f]) for f in files)))
            
        mfccs = extract_mfccs(scpfn, delta=delta, num_mfcc=num_mfcc, norm=normalize)
        logmfb = extract_logmfb(scpfn, delta=delta, num_mel=num_mfb, norm=normalize, energy=energy) 
        pitches = compute_pitch(scpfn) # ouput dim len*3
        
        feature_set={}
        for fname in mfccs:
            try:
                
                feature_set[fname] = np.hstack((pitches[fname],logmfb[fname], mfccs[fname]))
            except ValueError:
                print('calculate the min len')
                pitch_len = pitches[fname].shape[0]
                mfccs_len = mfccs[fname].shape[0]
                mfb_len = logmfb[fname].shape[0]
                
                logging.warning(
                    'dimension mismatch for file {}: {}, {}'
                    .format(fname, pitch_len, mfccs_len))
                length = min(pitch_len, mfccs_len, mfb_len)
                print('the diff is {}'.format(pitch_len - mfccs_len))
                feature_set[fname] = np.hstack((pitches[fname][:length],logmfb[fname][:length], mfccs[fname][:length]))
        return feature_set
    finally:
        tryremove(scpfn)

def compute_pitch(scpfile):
    logging.info('Extracting pitch')
    try:
        (outid, outfn) = tempfile.mkstemp()
        command_line = """compute-and-process-kaldi-pitch-feats scp:{} ark:{}""".format(scpfile, outfn)
        logging.info(command_line)
        subprocess.check_output(command_line, shell=True)
        pitches = {key:mat for key, mat in read_mat_ark(outfn)}
        logging.debug(pitches)
        return pitches
    finally:
        tryremove(outfn)
        
def extract_mfccs(scpfile, delta=0, num_mfcc=13, norm=True):
    logging.info('Extracting logmfbs')
    #TODO write configfile from args
    NCOEFF = num_mfcc
    if num_mfcc == 13:
        config = '' # use default value
    else:
        config = '--num-ceps='+str(NCOEFF) + ' --num-mel-bins='+str(NCOEFF)
    try:
        (outid_feat, outfn_feat) = tempfile.mkstemp()
        (outid_cmvn, outfn_cmvn) = tempfile.mkstemp()
        (outid_feat_norm, outfn_feat_norm) = tempfile.mkstemp()

        command_line = """compute-mfcc-feats {} scp:{} """.format(config, scpfile) + \
                                """ ark,scp:{},{}""".format(outfn_feat+'.ark', outfn_feat+'.scp')
        #print (command_line)
        logging.info(command_line)
        subprocess.check_output(command_line, shell=True)

        if not norm:
            logmfbs = { key:mat for key,mat in read_mat_ark(outfn_feat+'.ark') }
            logging.debug(logmfbs)
            return logmfbs
        else:
            command_cmvn = """compute-cmvn-stats scp:{} ark,scp:{},{}""".format(outfn_feat+'.scp', outfn_cmvn+'.ark', outfn_cmvn+'.scp')
            logging.info(command_cmvn)
            subprocess.check_output(command_cmvn, shell=True)
            #print (command_cmvn)
            command_apply_cmvn="""apply-cmvn scp:{} scp:{} ark:{}""".format(outfn_cmvn+'.scp',outfn_feat+'.scp', outfn_feat_norm)
            logging.info(command_apply_cmvn)
            subprocess.check_output(command_apply_cmvn, shell=True)
            #print (command_apply_cmvn)


            mfccs = { key:mat for key,mat in read_mat_ark(outfn_feat_norm) }
            logging.debug(mfccs)
            return mfccs

    finally:
        tryremove(outfn_feat+'.ark')
        tryremove(outfn_feat+'.scp')
        tryremove(outfn_cmvn+'.*')
        tryremove(outfn_feat_norm)

def extract_logmfb(scpfile, delta=0, num_mel=13, norm=True, energy=False):
    logging.info('Extracting logmfbs')
    #TODO write configfile from args
    NCOEFF = num_mel
    if not energy:
        config = '--num-mel-bins='+str(NCOEFF)
    else:
        config = '--num-mel-bins='+str(NCOEFF) + ' --use-energy=true'
    
    try:
        (outid_feat, outfn_feat) = tempfile.mkstemp()
        (outid_cmvn, outfn_cmvn) = tempfile.mkstemp()
        (outid_feat_norm, outfn_feat_norm) = tempfile.mkstemp()

        command_line = """compute-fbank-feats {} scp:{} """.format(config, scpfile) + \
                                """ ark,scp:{},{}""".format(outfn_feat+'.ark', outfn_feat+'.scp')
        #print (command_line)
        logging.info(command_line)
        subprocess.check_output(command_line, shell=True)

        if not norm:
            logmfbs = { key:mat for key,mat in read_mat_ark(outfn_feat+'.ark') }
            logging.debug(logmfbs)
            return logmfbs
        else:
            command_cmvn = """compute-cmvn-stats scp:{} ark,scp:{},{}""".format(outfn_feat+'.scp', outfn_cmvn+'.ark', outfn_cmvn+'.scp')
            logging.info(command_cmvn)
            subprocess.check_output(command_cmvn, shell=True)
            #print (command_cmvn)
            command_apply_cmvn="""apply-cmvn scp:{} scp:{} ark:{}""".format(outfn_cmvn+'.scp',outfn_feat+'.scp', outfn_feat_norm)
            logging.info(command_apply_cmvn)
            subprocess.check_output(command_apply_cmvn, shell=True)
            #print (command_apply_cmvn)

            logmfbs = { key:mat for key,mat in read_mat_ark(outfn_feat_norm) }
            logging.debug(logmfbs)
            return logmfbs

    finally:
        tryremove(outfn_feat+'.ark')
        tryremove(outfn_feat+'.scp')
        tryremove(outfn_cmvn+'.*')
        tryremove(outfn_feat_norm)

if __name__=='__main__':
    #run(input_wav_files, ouput_dir_path,  )
    run(input_wav_files, ouput_dir_path, batch_size=50, normalize=True, num_mfcc=40,
        num_mfb=40, energy=True)