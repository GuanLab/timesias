import os, sys
import numpy as np
import math


def annote_missing_features(data):
    """ Preprocess input feature

    params
    data

    yields
    processed_data
    
    """
    try:
        imax=data.shape[0]  # total number of time points
        jmax=data.shape[1]  # total number of features
    except:
        data=data.reshape((1,data.shape[0]))  # if no data exists
        imax=data.shape[0]
        jmax=data.shape[1]

    data=np.flip(data,axis=0)  # flip upside down

    processed_data=np.zeros((imax,jmax*2)) #
    
    i=0
    while (i<(data.shape[0])):
        j=0
        while (j<jmax):
            if (math.isnan(data[i][j])):
                processed_data[i][j*2]=-3000  # if feature at one timepoint is missing, set value to -3000
                processed_data[i][j*2+1]=0  # data is missing or not
            else:
                #print(data.shape[0],i)
                processed_data[i][j*2]=data[i][j] 
                processed_data[i][j*2+1]=1

            j=j+1
        i=i+1
    return processed_data



def construct_feature_matrix(data):
    """ Construct feature matrix
    params

        data: numpy array
        time-series input feature
    
    yields
        matrix: numpy array
        processed feature matrix

    """

    try:
        j=data.shape[1]  # total number of features 187*2 = 374
    except:
        data=data.reshape((1,data.shape[0]))

    i=data.shape[0]-1 # total number of features

    while(i<data.shape[0]):
        matrix=np.ones(36*j)*(-5000)   # array([-5000., -5000., -5000., ..., -5000., -5000., -5000.])  last 36 time points; default value: -5000

        if (i>=15):
            matrix[0:16*j]=data[i-15:i+1,:].flatten() # if record is longer than 16 time points, use features from the last 16 hours 
        else:
            matrix[(15-i)*j:16*j]=data[0:i+1,:].flatten() #
        
        # data normalization
        x_mean=np.nanmean(data,axis=0)
        x_std=np.nanstd(data[:,:],axis=0)+0.01
        #x_norm = np.nan_to_num((data[i,:] - x_mean) / x_std) #replace nan
        data_normed=(data[:,:]-x_mean)/x_std
        
        if (i>=15):
            matrix[16*j:32*j]=data_normed[i-15:i+1,:].flatten()
        else:
            matrix[(32*j-(i+1)*j):32*j]=data_normed[0:i+1,:].flatten()

        matrix[32*j:33*j]=x_std   # the 33 timepoint is the std of all previous timepoiints
        matrix[33*j:34*j]=np.sum(data[data ==-3000],axis=0)/(-3000.0)/float(data.shape[0])  # the 34 timepoint is the portion of missing timepoints
        
        baseline=[]
        jjj=0
        while (jjj<data.shape[1]):
            iii=0
            val=np.nan
            while (iii<data.shape[0]):
                if (data[iii][jjj]==-3000):
                    pass
                else:
                    if (math.isnan(val)):
                        val=data[iii][jjj]
                        timediff=data.shape[0]-iii
                iii=iii+1
            if (math.isnan(val)):
                baseline.append(np.nan)
                baseline.append(np.nan)
            else:
                baseline.append(val)
                baseline.append(timediff)
            jjj=jjj+1

        matrix[34*j:36*j]=np.asarray(baseline)
        i=i+1
    return matrix

