from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import sys

the_list=['<20','20~30','30~40','40~50','50~60','60~70','70~80','=80']
for the_file in the_list:
    y=np.genfromtxt(the_file,delimiter='\t')[:,1]
    pred=np.genfromtxt(the_file,delimiter='\t')[:,2]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    the_auc=metrics.auc(fpr, tpr)
    print(the_file,the_auc)

