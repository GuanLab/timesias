import lightgbm as lgb
import numpy as np

def lightgbm_train(train_data, train_label):
    """
    params:
    train_data: Numpy array
    train_label: Numpy array
    
    yields:
    gbm
    """
    lgb_train = lgb.Dataset(np.asarray(train_data), np.asarray(train_label))
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 150,
            'learning_rate': 0.05,
            'verbose': 0,
            'n_estimators': 400,
            'reg_alpha': 2.0,
            }

    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000
                )
    
    return gbm
    #print('Saving model...')
    #filename = 'finalized_model.sav.'+sys.argv[1]
    #pickle.dump(gbm, open(filename, 'wb'))

