from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import sys

the_list=['African_American.0','African_American.1','African_American.2','African_American.3','African_American.4','Asian.0','Asian.1','Asian.2','Asian.3','Asian.4','Caucasian.0','Caucasian.1','Caucasian.2','Caucasian.3','Caucasian.4']
for the_file in the_list:
    y=np.genfromtxt(the_file,delimiter='\t')[:,1]
    pred=np.genfromtxt(the_file,delimiter='\t')[:,2]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    the_auc=metrics.auc(fpr, tpr)
    print(the_file,the_auc)

