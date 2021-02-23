import os, sys
import numpy as np
import math


def annote_missing_features(data):
    """ Fill and Annote missing features and timepoints

    params
        data: an i x j Numpy array
            i is the total timepoints of records avaliable
            j is the total number of features

    yields
        processed_data: a i x 2j Numpy array
            for every feature, propagate an extra column to annote if it is missing (1/0)
    
    """
    try:
        imax=data.shape[0]  # total number of time points
        jmax=data.shape[1]  # total number of features
    except:
        data=data.reshape((1,data.shape[0]))  # if no data exists
        imax=data.shape[0]
        jmax=data.shape[1]

    data=np.flip(data,axis=0)  # flip upside down; last to earliest records

    processed_data=np.zeros((imax,jmax*2)) #
    
    i=0
    while (i<(data.shape[0])):
        j=0
        while (j<jmax):
            if (math.isnan(data[i][j])):
                processed_data[i][j*2]=-3000  # if the feature at one timepoint is missing, set value to -3000
                processed_data[i][j*2+1]=0  # data is missing or not
            else:
                #print(data.shape[0],i)
                processed_data[i][j*2]=data[i][j] 
                processed_data[i][j*2+1]=1

            j=j+1
        i=i+1
    return processed_data



def construct_feature_matrix(data, t=16):
    """ Construct feature matrix
    params
        data: numpy array
            time-series input feature
        t: cropped length of last records
            default: 16
            
    
    yields
        matrix: numpy array
            processed feature matrix
        f_names: list
            processed features
            [feature name]_[last_n_timepoint]_[val/ant]_[five types of features]
                [val/ant] represents:
                    1) val: feature value it self, if missing, replace with -3000
                    2) ant: missing value annotation, 1/0
                
                [fve types of features] represents:
                    1) norm: the above features normed by quantile 
                    2) std: feature-wise std
                    3) por: portion of missing values for each feature
                    4) b1: baseline 1: the earliest feature value for this feature 
                    5) b2: baseline 2: the earliest existing timepoint for this feature
    """

        def annote_missing_features(data):
        """ Fill and Annote missing features and timepoints

        params
            data: an i x j Numpy array
                i is the total timepoints of records avaliable
                j is the total number of features

        yields
            processed_data: a i x 2j Numpy array
                for every feature, propagate an extra column to annote if it is missing (1/0)

        """
        try:
            imax=data.shape[0]  # total number of time points
            jmax=data.shape[1]  # total number of features
        except:
            data=data.reshape((1,data.shape[0]))  # if no data exists
            imax=data.shape[0]
            jmax=data.shape[1]

        data=np.flip(data,axis=0)  # flip upside down; last to earliest records

        processed_data=np.zeros((imax,jmax*2)) #

        i=0
        while (i<(data.shape[0])):
            j=0
            while (j<jmax):
                if (math.isnan(data[i][j])):
                    processed_data[i][j*2]=-3000  # if the feature at one timepoint is missing, set value to -3000
                    processed_data[i][j*2+1]=0  # data is missing or not
                else:
                    #print(data.shape[0],i)
                    processed_data[i][j*2]=data[i][j]
                    processed_data[i][j*2+1]=1

                j=j+1
            i=i+1
        return processed_data

    
    data = annote_missing_features(data)

    try:
        j=data.shape[1]  # total number of features 187*2 = 374
    except:
        data=data.reshape((1,data.shape[0]))

    i=data.shape[0]-1 # total number of features

    while(i<data.shape[0]):
        matrix=np.ones((2*t+4)*j)*(-5000)   # array([-5000., -5000., -5000., ..., -5000., -5000., -5000.])  total length; default value: -5000

        if (i>=t-1):
            matrix[0:t*j]=data[i+1-t:i+1,:].flatten() # if record is longer than t (16) time points, use features from the last t length 
        else:
            matrix[(t-1-i)*j:t*j]=data[0:i+1,:].flatten() 
        
        # data normalization
        x_mean=np.nanmean(data,axis=0)
        x_std=np.nanstd(data[:,:],axis=0)+0.01
        #x_norm = np.nan_to_num((data[i,:] - x_mean) / x_std) #replace nan
        data_normed=(data[:,:]-x_mean)/x_std
        
        if (i>=t-1):    # t-2*t are normed features values by quantile
            matrix[t*j:t*2*j]=data_normed[i+1-t:i+1,:].flatten()
        else:
            matrix[(t*2*j-(i+1)*j):t*2*j]=data_normed[0:i+1,:].flatten()

        matrix[t*2*j:(t*2+1)*j]=x_std   # the 33 timepoint is the std of all previous timepoiints
        matrix[(t*2+1)*j:(t*2+2)*j]=np.sum(data[data ==-3000],axis=0)/(-3000.0)/float(data.shape[0])  # the 34 timepoint is the portion of missing timepoints
        
        # baseline feature: 
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
                        timediff=data.shape[0]-iii  # earliest timepoint for this feature to be collected
                iii=iii+1
            if (math.isnan(val)):
                baseline.append(np.nan)
                baseline.append(np.nan)
            else:
                baseline.append(val)  #earliest timepoint feature value
                baseline.append(timediff)  #earliest timepoint
            jjj=jjj+1

        matrix[(t*2+2)*j:(t*2+4)*j]=np.asarray(baseline)
        i=i+1

    return matrix

