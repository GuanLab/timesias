import sys, os, argparse, textwrap
from common import *

def main():
    parser = argparse.ArgumentParser(description = 'Predict outcome from time-series data',
            usage = 'use "python %(prog)s --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-g','--gs_file_path', type=  str,
            help = 'path to gold-standards and file path file')

    args = parser.parse_args()
    
    
    opts = vars(args)
    run(**opts)



def run(gs_file_path):
    five_fold_cv(gs_file_path)

    # age-specific evaluation
    #the_list=['<20','20~30','30~40','40~50','50~60','60~70','70~80','=80']



if __name__ == '__main__':
    main()
