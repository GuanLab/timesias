from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import sys

the_list=['0','1','2','3','4']
y_long=np.zeros(0)
pred_long=np.zeros(0)
for the_id in the_list:
    y=np.genfromtxt((sys.argv[1]+'.'+the_id),delimiter='\t')[:,1]
    pred=np.genfromtxt((sys.argv[1]+'.'+the_id))[:,2]
    y_long=np.hstack((y,y_long))
    pred_long=np.hstack((pred,pred_long))


a=np.arange(len(y_long))

F=open(('auc_bootstrap.txt.'+sys.argv[1]),'w')
i =0
while (i<10000):
    ll = np.random.choice(a, size=a.shape, replace=True)
    y_tmp=y_long[ll]
    pred_tmp=pred_long[ll]
    fpr, tpr, thresholds = metrics.roc_curve(y_tmp, pred_tmp, pos_label=1)
    the_auc=metrics.auc(fpr, tpr)
    F.write('%.4f\n' % the_auc)
    i=i+1
F.close()

