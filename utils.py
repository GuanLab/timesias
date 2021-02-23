import os, sys
import numpy as np
import math

def construct_feature_matrix(data, t, f):
    """ Construct feature matrix
    params
        data: numpy array
            time-series input feature
        t: cropped length of last records
            default: 16
        f: list
            features to be used
            e.g. ['norm', 'std', 'missing_portion', 'baseline']
    
    yields
        matrix: numpy array
            processed feature matrix
        f_names: list
            processed features
            [feature name]_[val/ant]_[last_nth_timepoint]_[ori/norm]
            [feature name]_[val/ant]_[four features-wise information]
                [val/ant] represents:
                    1) val: feature value it self, if missing, replace with -3000
                    2) ant: missing value annotation, 1/0
                [ori/norm] represents the following information for last t time points:
                    1) ori: original value of the above, if missing, replace with -5000
                    2) norm: normed by quantile of the above
                [four feature-wise information] represents the following feature-wise information:
                    2) std: feature-wise std
                    3) mp: portion of missing values for each feature 
                    4) bs: for val,  the earliest feature value for this feature; 
                           for ant, the earliest existing timepoint for this feature
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

    def last_i(d, i, t):
        """ selecg last t timepoints out of i
        """
        j = d.shape[1]
        m=np.ones(t*j)*(-5000)  # array([-5000., -5000., -5000., ..., -5000., -5000., -5000.]) default value: -5000
        if (i>=t-1):   # if record is longer than t (16) time points, use features from the last t length 
            m[0:t*j]=d[(i+1-t):(i+1),:].flatten()
        else:  # if the record is shorter than t time points, the shorted timepoints are replaced with -5000
            m[(t-1-i)*j:t*j]=d[0:(i+1),:].flatten()
        
        return m

    def norm_features(d):
        x_mean=np.nanmean(d,axis=0)
        x_std=np.nanstd(d,axis=0)+0.01
        new_d=(d-x_mean)/x_std
        return new_d,  x_std, x_mean

    def missing_portion(d):
        m = np.ones(d.shape[1])*(-5000)
        m[:] = np.sum(d[d ==-3000],axis=0)/(-3000.0)/float(d.shape[0]) 
        return m

    def baseline(d):
        b=[]
        j=0
        while (j<d.shape[1]):
            i=0
            val=np.nan
            while (i<d.shape[0]):
                if (d[i][j]==-3000):
                    pass
                else:
                    if (math.isnan(val)):
                        val=d[i][j]
                        timediff=d.shape[0]-i  # earliest timepoint for this feature to be collected
                i=i+1
            if (math.isnan(val)):
                b.append(np.nan)
                b.append(np.nan)
            else:
                b.append(val)  #earliest timepoint feature value
                b.append(timediff)  #earliest timepoint
            j=j+1
        b = np.asarray(b)
        #b = np.reshape(b, (1, b.shape[0]))
        return b

    f_names = data.columns
    
    data =  np.array(data)
    data = annote_missing_features(data)
    f_names = [f+a for f in f_names for a in ['_val', '_ant']]

    #try:
    #    j=data.shape[1]  # total number of features 187*2 = 374, extend two fold
    #except:
    #    data=data.reshape((1,data.shape[0]))

    
    new_t = t # total_length
    if 'norm' in f:
        new_t += t
    if 'std' in f:
        new_t +=1
    if 'missing_portion' in f:
        new_t +=1
    if 'baseline' in f:
        new_t +=2

    i=data.shape[0]-1 # total number of time points
    #print(data.shape)
    m = []
    f = []
    while(i<data.shape[0]):  # there is no loop! just the last timepoint
        #print(i)
        new_m  = last_i(data, i, t)
        m.append(new_m)
        new_f = [fn+'_'+str(a)+'_ori' for a in range(t)for fn in f_names]
        f.extend(new_f)
        #print(new_f)

        if 'norm' in f:
            d_n, std, mean = norm_features(data)
            new_m  = last_i(d_n, i, t)
            #print(new_m.shape)
            m.append(new_m)
            new_f = [fn+'_'+str(a)+'_norm' for a in range(t)for fn in f_names]
            f.extend(new_f)

        if 'std' in f:
            d_n, std, mean = norm_features(data)
            #print(std.shape)
            m.append(std)
            new_f = [fn+'_std' for fn in f_names]
            f.extend(new_f)

        if 'missing_portion' in f:
            new_m = missing_portion(data)
            #print(new_m.shape)
            m.append(new_m)
            new_f = [fn+'_mp' for fn in f_names]
            f.extend(new_f)

        if 'baseline' in f:
            new_m = baseline(data)
            #print(new_m.shape)
            m.append(new_m)
            new_f = [fn+'_bs' for fn in f_names]
            f.extend(new_f)

        i=i+1
    
    m = np.concatenate(m, axis = 0)
    matrix = np.reshape(m, (1, m.shape[0])) 
    #print(matrix.shape)

    return matrix, f

