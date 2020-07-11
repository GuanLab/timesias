from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import sys

y=np.genfromtxt(('test_gs.dat.'+sys.argv[1]),delimiter=',')[:,1]
pred=np.loadtxt(('prediction.dat.'+sys.argv[1]))

print(y.shape)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
the_auc=metrics.auc(fpr, tpr)
F=open('auc.txt','w')
F.write('%.4f\n' % the_auc)
F.close()

precision, recall, thresholds = precision_recall_curve(y, pred)
the_auprc = auc(recall, precision)
F=open('auprc.txt','w')
F.write('%.4f\n' % the_auprc)
F.close()
