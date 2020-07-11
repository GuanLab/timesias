import sys
import preprocess
import numpy as np
import os, shutil, zipfile
import glob
import pickle


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
    whole_train= load_challenge_data(('data/'+table[0]+'.psv'))

    data=whole_train
    processed_data=preprocess.preprocess(data)
    matrix=preprocess.feature(processed_data)
    train_array.append(matrix)
        
train_array=np.asarray(train_array)
value=est.predict(train_array)
np.savetxt(('prediction.dat.'+sys.argv[1]),value)
