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
from collections import defaultdict


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
        assert os.path.exists(fpath), "File '"+fpath+"' not exist!" 
        d=pd.read_csv(fpath, sep = '|', header = 0)
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

def five_fold_cv(gs_filepath, n, f, shap):
    """
    params
        gs_filepath
        n
        f
    yields
        
    """
    gs_file = pd.read_csv(gs_filepath, header = None)
    f_path = gs_file[0].to_list()
    f_gs = gs_file[1].to_list()
    kf = KFold(n_splits=5, random_state= 0, shuffle= True)
    out_eva = open('eva.tsv', 'w')
    out_eva.write("%s\t%s\t%s\n" %('fold', 'AUROC', 'AUPRC'))
    
    if shap:
        all_feature_shap = []
        all_t_shap = []
    
    for i,(train_idx, test_idx) in enumerate(kf.split(f_path)):
        print("Start fold "+str(i)+" in five-fold cross-validation ..")
        # load train
        print("Load training data ...")
        train_f = [f_path[j] for j in train_idx]
        train_matrix, _ = load_data(train_f, n, f)
        train_gs = [f_gs[j] for j in train_idx]
        # train model
        print("Start training ...")
        gbm = lightgbm_train(train_matrix, train_gs)
        
        os.makedirs('./models', exist_ok = True)
        filename = './models/finalized_model.sav.'+str(i)
        print('Saving model to '+ filename+ '...') # save model to file
        pickle.dump(gbm, open(filename, 'wb'))
        
        # load test
        print("Load test data ...")
        test_f = [f_path[j] for j in test_idx]
        test_matrix,f_names = load_data(test_f, n, f)
        test_gs = [f_gs[j] for j in test_idx]
        
        # test model
        print("Start evaludation ...")
        test_pred = gbm.predict(test_matrix)

        # evaluation
        the_auc, the_auprc = evaluation(test_gs, test_pred)

        print("AUC:%.4f; AUPRC: %.4f" % (the_auc, the_auprc))
        out_eva.write("%d\t%.4f\t%.4f\n" %(i, the_auc, the_auprc))

        # SHAP
        if shap:
            print("Start SHAP analysis ...")
            feature_shap, t_shap = shap_analysis(gbm, test_matrix, f_names)
            feature_shap['fold'] = i
            t_shap['fold'] = i
            all_feature_shap.append(feature_shap)
            all_t_shap.append(t_shap)

    out_eva.close()

    if shap:
        all_feature_shap = pd.concat(all_feature_shap)
        all_t_shap = pd.concat(all_t_shap)
        all_feature_shap.to_csv('shap_group_by_measurment.csv', index = False)
        all_t_shap.to_csv('shap_group_by_timeslot.csv', index = False)


def specific_evaluation(gs_filepath, n, f):
    """ Conduct specific evaludation and shap analysis at the specified dataset
    Params
        gs_filepath
        n
        f
    Yields
    
    """
    gs_file = pd.read_csv(gs_filepath, header = None)
    f_path = gs_file[0].to_list()
    f_gs = gs_file[1].to_list()
    test_idx = [55,73,75,78,86,92,93,95]# range(len(f_path))
    print(test_idx)
    model_paths = glob('./models/finalized_model.sav.*')
    for p in model_paths:
        gbm = pickle.load(open(p, 'rb'))
        # load test
        test_f = [f_path[j] for j in test_idx]

        test_matrix, f_names = load_data(test_f, n, f)
        test_gs = [f_gs[j] for j in test_idx]

        test_pred =gbm.predict(test_matrix)

        # evaluation
        the_auc, the_auprc = evaluation(test_gs, test_pred)
        
        print(the_auc, the_auprc)
        
        # SHAP analysis
        feature_shap, t_shap = shap_analysis(gbm, test_matrix, f_names)
    
    feature_shap.to_csv('shap_group_by_measurment.csv', index= False)
    t_shap.to_csv('shap_group_by_timeslot.csv', index = False)


def shap_analysis(regressor, Test_X, f_names):
    """ SHAP analysis on a sspecific dataset
    Params
        regressor
        df
    Yields

    """
    shap_values = shap.TreeExplainer(regressor).shap_values(Test_X)
    all_f_dict = defaultdict(lambda:[])
    for i, n in enumerate(f_names):
        all_f_dict[n.split('|')[0]].append(i)  # get all unique features
    
    #print(all_f_dict)
    time_p_dict = defaultdict(lambda:[])
    for i, n in enumerate(f_names):
        if n.endswith('ori') or n.endswith('norm'):
            time_p_dict[n.split('|')[2]].append(i)  # ge all unique timepoints
    
    #print(time_p_dict)
    # combine subfeatures for each time slots
    feature_shap = {"feature":[], "mean|SHAP val|":[]}
    for k, v in all_f_dict.items():
        feature_shap["feature"].append(k)
        feature_shap["mean|SHAP val|"].append(abs(shap_values[:, v].sum(axis = 1)).mean())
    feature_shap = pd.DataFrame.from_dict(feature_shap)
    #.sort_values(by="mean|SHAP val|", ascending = False).set_index('feature')
    
    # combine subfeatures for each feature types
    t_shap = {"the_last_nth_timepoint":[], "mean|SHAP val|":[]}
    for k, v in time_p_dict.items():
        t_shap["the_last_nth_timepoint"].append(k)
        t_shap["mean|SHAP val|"].append(abs(shap_values[:, v].sum(axis = 1)).mean())
    t_shap = pd.DataFrame.from_dict(t_shap)
    #sort_values(by="mean|SHAP val|", ascending = False).set_index('feature')
    
    return feature_shap, t_shap

def plot_shap(feature_shap = None, t_shap = None):
    """ Plot SHAP

    Params
        df: columns: 

    Yields
        plots
    """
    from bokeh.io import output_file, show
    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Spectral5
    from bokeh.plotting import figure
    from bokeh.layouts import row, column
    from bokeh.transform import factor_cmap, jitter

    output_file('top_feature_report.html')
    
    feature_shap = pd.read_csv('shap_group_by_measurment.csv')
    group_feature = feature_shap.groupby("feature")
    group_feature_sorted = group_feature['mean|SHAP val|'].agg('mean').sort_values(ascending=False).index.to_list()[:50]

    source = ColumnDataSource(group_feature)
    source_2 = ColumnDataSource(feature_shap)
    p1 = figure(plot_height=400, x_range=group_feature_sorted, title="Top 50 measurements", toolbar_location=None, tools="")
    p1.vbar(x='feature', top='mean|SHAP val|_mean', width=0.9, alpha=0.5, source=source)
    p1.circle(x='feature', y = 'mean|SHAP val|',size = 3, source=source_2, alpha=0.5)
    p1.y_range.start = 0
    
    p1.xgrid.grid_line_color = None
    p1.xaxis.axis_label = "measurement"
    p1.xaxis.major_label_orientation = "vertical"
    p1.outline_line_color = None

    t_shap = pd.read_csv('shap_group_by_timeslot.csv')
    t_shap.the_last_nth_timepoint = t_shap.the_last_nth_timepoint.astype(str)
    group_t = t_shap.groupby("the_last_nth_timepoint")
    group_t_sorted = group_t['mean|SHAP val|'].agg('mean').sort_values(ascending=False).index.to_list()
    source = ColumnDataSource(group_t)
    source_2 = ColumnDataSource(t_shap)
    p2 = figure(plot_height=200, x_range=group_t_sorted, title="Top last timepoints", toolbar_location=None, tools="")
    p2.vbar(x='the_last_nth_timepoint', top='mean|SHAP val|_mean', width=0.9, alpha=0.5, source=source)
    p2.circle(x='the_last_nth_timepoint', y = 'mean|SHAP val|',size = 3, source=source_2, alpha=0.5)
    p2.y_range.start = 0

    p2.xgrid.grid_line_color = None
    p2.xaxis.axis_label = "the_last_nth_timepoint"
    p2.xaxis.major_label_orientation = "vertical"
    p2.outline_line_color = None
    
    p = column(p1,p2)
    show(p)

plot_shap()












