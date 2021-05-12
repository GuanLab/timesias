from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd
import numpy as np
def pearsonr_cor(pred, label):
    """ Return Pearson's correlation between prediction and label
    """
    cor, _ = pearsonr(pred, label)
    return cor

def spearmanr_cor(pred, label):
    """ Return Spearman's correlation between prediction and label
    """
    cor, _ = spearmanr(pred, label)
    return cor

def compute_auroc(pred, label):
    """ Calculate AUROC of predictions in terms of label
    """
    #label = np.array(label)
    #pred = np.array(pred)
    fpr, tpr, thresholds = roc_curve(label, pred, pos_label =1)
    auroc = auc(fpr, tpr)
    return auroc

def compute_auprc(pred, label):
    """ Calculate AUPRC of predictions in terms of label
    """
    #label = np.array(label)
    #pred = np.array(pred)
    precision, recall, thresholds = precision_recall_curve(label, pred)
    auprc = auc(recall, precision)
    return auprc


def c_index(pred, label):
    """ Compute C-idex of the predictions in terms of ground truth
    
    Parameters:
    -----------
    pred: list
        prediction
    label: list
        ground truth
    pred and label are the same length
    
    Yields:
    -------
    cidx: float
        C-index (between 0 to 1)
    """
    from itertools import permutations
    pred = list(pred)
    label = list(label)
    perm = permutations(list(range(len(pred))), 2)
    survive = 0
    total = 0
    for i, j in perm:
        if label[i]<label[j]:
            total +=1
            if pred[i]<pred[j]:
                survive += 1
    cidx = survive/total
    return cidx

def boostrapping_confidence_interval(pred_all, gs_all, eva_func, ci):
    """ Boostrapping to get a 95 confidence interval for prediction performance
    Parameters:
    -----------
    pred_all: list
        all predictions from k-fold cross-validations
    gs_all: list
        all gold standards from k-fold cross-validations
    eva_func: function
        evaludation function
    ci: confidence interval

    Yields:
    -------
    mb: float
        middle bound
    lb: float
        lower bound
    ub: float
        upper bound
    """
    import numpy as np
    import random
    # set random seed
    random.seed(0)

    # prediction-groundtruth pairs from all five fold cross validation
    tmp = np.array([pred_all, gs_all]).T
    # calculate overall correlation
    mb = eva_func(tmp[:,0], tmp[:,1])
    # start boostrapping ...
    eva_all = []
    for i in range(100):
        tmp_new = random.choices(tmp, k = len(tmp))
        tmp_new = np.array(tmp_new)
        eva = eva_func(tmp_new[:,0], tmp_new[:,1])
        eva_all.append(eva)
    eva_all = sorted(eva_all)
    #print(eva_all)
    lb = eva_all[round(100*(0.5-ci*0.5))]
    ub = eva_all[round(100*(0.5+ci*0.5))]
    return mb, lb, ub
