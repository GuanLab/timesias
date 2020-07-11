from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import sys

the_list=['Female.0','Female.1','Female.2','Female.3','Female.4','Male.0','Male.1','Male.2','Male.3','Male.4']
for the_file in the_list:
    y=np.genfromtxt(the_file,delimiter='\t')[:,1]
    pred=np.genfromtxt(the_file,delimiter='\t')[:,2]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    the_auc=metrics.auc(fpr, tpr)
    print(the_file,the_auc)

