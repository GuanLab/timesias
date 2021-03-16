import sys, os, argparse, textwrap
from common import *

def main():
    parser = argparse.ArgumentParser(description = 'Predict outcome from time-series data',
            usage = 'use "python %(prog)s --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-g','--gs_file_path', type=str,
            help = 'path to gold-standards and file path file')

    parser.add_argument('-t','--last_n_records', type=int,
            help = 'Use last n records. defulat: 16',
            default =  16)

    parser.add_argument('-f','--extra_features', type=str,
            default = ['norm', 'std', 'missing_portion', 'baseline'],
            help = '''
            Which extra features to use.
            default: ['norm', 'std', 'missing_portion', 'baseline']
            ''')
    parser.add_argument('--shap',
            action = 'store_true',
            help = 'Conduct shap analysis on test set')

    args = parser.parse_args()
    
    
    opts = vars(args)
    run(**opts)


def run(gs_file_path, last_n_records, extra_features, shap):
    five_fold_cv(gs_file_path, last_n_records, extra_features, shap)
    #specific_evaluation(gs_file_path, last_n_records, extra_features)

if __name__ == '__main__':
    main()
