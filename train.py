import numpy as np
import sklearn
import sklearn.ensemble
import os
import pickle
import lightgbm as lgb
import preprocess
import sys

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    return data

FILE=open(('train_gs.dat.'+sys.argv[1]),'r')
train_array=[]
train_gs=[]
for line in FILE:
    line=line.strip()
    table=line.split(',')
#    whole_train=np.loadtxt(table[0])
    whole_train= load_challenge_data(('data/'+table[0]+'.psv'))
    train_gs.append(float(table[1]))
    data=whole_train
    processed_data=preprocess.preprocess(data)
    matrix=preprocess.feature(processed_data)
    train_array.append(matrix)
        

lgb_train = lgb.Dataset(np.asarray(train_array), np.asarray(train_gs))

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 150,
    'learning_rate': 0.05,
    'verbose': 0,
    'n_estimators': 400,
    'reg_alpha': 2.0,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000
                )

print('Saving model...')
# save model to file

filename = 'finalized_model.sav.'+sys.argv[1]
pickle.dump(gbm, open(filename, 'wb'))

