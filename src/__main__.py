import sys, os, argparse, textwrap
from .common import *

def main():
    parser = argparse.ArgumentParser(description = 'Predict outcomes from time-series data',
            usage = 'use "python %(prog)s --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-g','--gs_file_path', type=str,
            help = 'path to the gold-standard file, for example, ./data/gs.file')

    parser.add_argument('-t','--last_n_records', type=int,
            help = 'Use last n records. default: 16',
            default =  16)

    parser.add_argument('-f','--extra_features', type=str,
            nargs = '*',
            default = ['norm', 'std', 'missing_portion', 'baseline'],
            help = '''
            Which extra features to use.
            default: ['norm', 'std', 'missing_portion', 'baseline']
            ''')
    parser.add_argument('--shap',
            action = 'store_true',
            help = 'Conduct shap analysis on the test set')

    args = parser.parse_args()
    
    if args.gs_file_path is not None:
        print('Gold-standard file path: '+args.gs_file_path)
        if os.path.isfile(args.gs_file_path):
            pass
        else:
            sys.exit("Gold standard file doesn't exist!")

    if args.last_n_records is not None:
        print('Use Last '+str(args.last_n_records)+' records for predicton.')
    if len(args.extra_features) >0:
        print('Use extra features: '+','.join(args.extra_features)+'.')
    if args.shap == True:
        print('Perform SHAP analysis after model training.')

    opts = vars(args)
    run(**opts)


def run(gs_file_path, last_n_records, extra_features, shap):
    five_fold_cv(gs_file_path, last_n_records, extra_features, shap)
    #specific_evaluation(gs_file_path, last_n_records, extra_features)

if __name__ == '__main__':
    main()
