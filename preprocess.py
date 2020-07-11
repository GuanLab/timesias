#!/usr/bin/perl
#

import numpy as np
import math
def preprocess(data):
#    print(data.shape)
    try:
        imax=data.shape[0]
        jmax=data.shape[1]
    except:
        data=data.reshape((1,data.shape[0]))
        imax=data.shape[0]
        jmax=data.shape[1]
    
    data=np.flip(data,axis=0)

    processed_data=np.zeros((imax,jmax*2))
    i=0
    while (i<(data.shape[0])):
        j=0
        while (j<jmax):
            if (math.isnan(data[i][j])):
                processed_data[i][j*2]=-3000
            else:
                print(data.shape[0],i)
                processed_data[i][j*2]=data[i][j]
            if (math.isnan(data[i][j])):
                processed_data[i][j*2+1]=0
            else:
                processed_data[i][j*2+1]=1
            j=j+1
        i=i+1
    return processed_data
	

def feature(whole_train):
    try:
        j=whole_train.shape[1]
    except:
        whole_train=whole_train.reshape((1,whole_train.shape[0]))
    
    i=whole_train.shape[0]-1
    
    while(i<whole_train.shape[0]):
        matrix=np.ones(13464)*(-5000)
        
        if (i>=15): 
            matrix[0:5984]=whole_train[i-15:i+1,:].flatten()
        else:
            matrix[(5984-(i+1)*374):5984]=whole_train[0:i+1,:].flatten()

        x_mean=np.nanmean(whole_train[:,:],axis=0)
        x_std=np.nanstd(whole_train[:,:],axis=0)+0.01
        x_norm = np.nan_to_num((whole_train[i,:] - x_mean) / x_std)
        whole_train_normed=(whole_train[:,:]-x_mean)/x_std
        if (i>=15): 
            matrix[5984:11968]=whole_train_normed[i-15:i+1,:].flatten()
        else:
            matrix[(11968-(i+1)*374):11968]=whole_train_normed[0:i+1,:].flatten()

        matrix[11968:12342]=x_std
        matrix[12342:12716]=np.sum(whole_train[:,:][whole_train[:,:]==-3000],axis=0)/(-3000.0)/float(whole_train.shape[0])
        baseline=[]
        jjj=0
        while (jjj<whole_train.shape[1]):
            iii=0
            val=np.nan
            while (iii<whole_train.shape[0]):
                if (whole_train[iii][jjj]==-3000):
                    pass
                else:
                    if (math.isnan(val)):
                        val=whole_train[iii][jjj]
                        timediff=whole_train.shape[0]-iii
                iii=iii+1
            if (math.isnan(val)):
                baseline.append(np.nan)
                baseline.append(np.nan)
            else:
                baseline.append(val)
                baseline.append(timediff)
            jjj=jjj+1

        matrix[12716:13464]=np.asarray(baseline)
        i=i+1
    return matrix
