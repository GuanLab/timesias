import sys
import preprocess
import numpy as np
import os, shutil, zipfile
import glob
import pickle

import shap

est=pickle.load(open(('finalized_model.sav.'+sys.argv[1]), 'rb'))

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    return data

FILE=open(('test_gs.dat.'+sys.argv[1]),'r')
train_array=[]
for line in FILE:
    line=line.strip()
    table=line.split(',')
#    whole_train=np.loadtxt(table[0])
    whole_train= load_challenge_data(('../../preprocess_code/physionet_format/'+table[0]+'.psv'))

    data=whole_train
    processed_data=preprocess.preprocess(data)
    matrix=preprocess.feature(processed_data)
    train_array.append(matrix)
        
train_array=np.asarray(train_array)
explainer=shap.TreeExplainer(est)
shap_values=explainer.shap_values(train_array)
y_array=est.predict(train_array)
shap_mean = np.mean(shap_values*((np.array(y_array)-0.5)*2).reshape(-1,1), axis=0)


FILE=open('../../preprocess_code/physionet_format/A100001.psv')
titleline=FILE.readline()
titleline=titleline.strip()
titleline=titleline.replace(' ','_')
titles=titleline.split('|')

SHAP=open(('shap.txt.'+sys.argv[1]),'w')
i_shap=0
i=0
while (i<8):
    for the_title in titles:
        SHAP.write('last_'+str(i)+'_'+the_title+'\t'+str(shap_mean[i_shap]))
        SHAP.write('\n')
        i_shap=i_shap+1
        SHAP.write('last_'+str(i)+'_'+the_title+' missing'+'\t'+str(shap_mean[i_shap]))
        SHAP.write('\n')
        i_shap=i_shap+1
    i=i+1
i=0
while (i<8):
    for the_title in titles:
        SHAP.write('last_'+str(i)+'_'+the_title+'_normed'+'\t'+str(shap_mean[i_shap]))
        SHAP.write('\n')
        i_shap=i_shap+1
        SHAP.write('last_'+str(i)+'_'+the_title+'_normed missing'+'\t'+str(shap_mean[i_shap]))
        SHAP.write('\n')
        i_shap=i_shap+1
    i=i+1
for the_title in titles:
    SHAP.write(the_title+'_std'+'\t'+str(shap_mean[i_shap]))
    SHAP.write('\n')
    i_shap=i_shap+1
for the_title in titles:
    SHAP.write(the_title+'_mean'+'\t'+str(shap_mean[i_shap]))
    SHAP.write('\n')
    i_shap=i_shap+1
for the_title in titles:
    i=0
    while (i<8):
        SHAP.write(the_title+'_baseline'+'\t'+str(shap_mean[i_shap]))
        SHAP.write('\n')
        i_shap=i_shap+1
        SHAP.write(the_title+'_timelapse'+'\t'+str(shap_mean[i_shap]))
        SHAP.write('\n')
        i_shap=i_shap+1
        i=i+1
