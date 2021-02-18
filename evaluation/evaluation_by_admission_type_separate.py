from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import sys

the_list=['Elective.0','Elective.1','Elective.2','Elective.3','Elective.4','Emergency.0','Emergency.1','Emergency.2','Emergency.3','Emergency.4','Trauma_Center.0','Trauma_Center.1','Trauma_Center.2','Trauma_Center.3','Trauma_Center.4','Urgent.0','Urgent.1','Urgent.2','Urgent.3','Urgent.4','Others_unknown.0','Others_unknown.1','Others_unknown.2','Others_unknown.3','Others_unknown.4']
for the_file in the_list:
    y=np.genfromtxt(the_file,delimiter='\t')[:,1]
    pred=np.genfromtxt(the_file,delimiter='\t')[:,2]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    the_auc=metrics.auc(fpr, tpr)
    print(the_file,the_auc)

