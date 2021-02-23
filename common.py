import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
from model import lightgbm_train
from glob import glob
from utils import *
import shap


def load_data(all_fpath, n, f):
    """
    params
        all_fpath: a list of path of all sample files
        n: int
            last n records used for prediction
        f: list
            extra features

    yields
        all_matrix: Numpy array
            all feature matrix of all files in all_fpath
        fnames: list
            featrue names
    """
    all_matrix =[]
    for fpath in all_fpath:
        new_fpath = './data/'+fpath
        assert os.path.exists(new_fpath), "File '"+new_fpath+"' not exist!" 
        d=pd.read_csv(new_fpath, sep = '|', header = 0)
        m, fnames = construct_feature_matrix(d, n, f)
        all_matrix.append(m)
    all_matrix=np.concatenate(all_matrix, axis = 0)
    return all_matrix, fnames

def evaluation(gs,pred):
    """
    params
        gs
        pred
    
    yields
        the_auc
        the_auprc
    """
    #auc
    fpr, tpr, thresholds = metrics.roc_curve(gs, pred, pos_label=1)
    the_auc=metrics.auc(fpr, tpr)
    #aurpc
    precision, recall, thresholds = precision_recall_curve(gs, pred)
    the_auprc = metrics.auc(recall, precision)

    return the_auc, the_auprc

def five_fold_cv(gs_filepath, n, f):
    """
    params
        gs_filepath
        n
        f
    yields
        
    """

    #gs_filepath = './data/gs.file'
    gs_file = pd.read_csv(gs_filepath, header = None)
    f_path = gs_file[0].to_list()
    f_gs = gs_file[1].to_list()
    #print(f_path, f_gs)
    kf = KFold(n_splits=5, random_state= 0, shuffle= True)
    out_eva = open('eva.tsv', 'w')
    out_eva.write("%s\t%s\t%s\n" %('fold', 'AUROC', 'AUPRC'))
    for i,(train_idx, test_idx) in enumerate(kf.split(f_path)):

        # load train
        train_f = [f_path[j] for j in train_idx]
        train_matrix, _ = load_data(train_f, n, f)
        train_gs = [f_gs[j] for j in train_idx]
        # train model
        gbm = lightgbm_train(train_matrix, train_gs)
        
        os.makedirs('./models', exist_ok = True)
        filename = './models/finalized_model.sav.'+str(i)
        print('Saving model to '+ filename+ '...') # save model to file
        pickle.dump(gbm, open(filename, 'wb'))
        
        # load test
        test_f = [f_path[j] for j in test_idx]
        test_matrix,f_names = load_data(test_f, n, f)
        test_gs = [f_gs[j] for j in test_idx]
        
        test_pred =gbm.predict(test_matrix)

        # evaluation
        the_auc, the_auprc = evaluation(test_gs, test_pred)

        print(the_auc, the_auprc)
        out_eva.write("%d\t%.4f\t%.4f\n" %(i, the_auc, the_auprc))
    out_eva.close()

def specific_evaluation(test_idx):
    """
        gs_filepath, n, f
    """
    f_models = glob('./models/finalized_model.sav.*')
    for f in f_models:
        gbm = pickle.load(open(f, 'rb'))
        # load test
        test_f = [f_path[j] for j in test_idx]

        test_matrix, f_names = load_data(test_f, n, f)
        test_gs = [f_gs[j] for j in test_idx]
    
        test_pred =gbm.predict(test_matrix)

        # evaluation
        the_auc, the_auprc = evaluation(test_gs, test_pred)

        print(the_auc, the_auprc)

def shap_analysis(regressor, Test_X, f_names):
    """ SHAP analysis on a sspecific dataset
    params
        regressor
        df
    yields

    """
    shap_values = shap.TreeExplainer(regressor).shap_values(Test_X)


    pass





