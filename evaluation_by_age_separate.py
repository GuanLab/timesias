from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import sys

the_list=['<20.0','<20.1','<20.2','<20.3','<20.4','20~30.0','20~30.1','20~30.2','20~30.3','20~30.4','30~40.0','30~40.1','30~40.2','30~40.3','30~40.4','40~50.0','40~50.1','40~50.2','40~50.3','40~50.4','50~60.0','50~60.1','50~60.2','50~60.3','50~60.4','60~70.0','60~70.1','60~70.2','60~70.3','60~70.4','70~80.0','70~80.1','70~80.2','70~80.3','70~80.4','=80.0','=80.1','=80.2','=80.3','=80.4']
for the_file in the_list:
    y=np.genfromtxt(the_file,delimiter='\t')[:,1]
    pred=np.genfromtxt(the_file,delimiter='\t')[:,2]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    the_auc=metrics.auc(fpr, tpr)
    print(the_file,the_auc)

