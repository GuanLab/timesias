import sys, os, argparse, textwrap
from common import *

def main():
    parser = argparse.ArgumentParser(description = 'Predict outcome from time-series data',
            usage = 'use "python %(prog)s --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-g','--gs_file_path', type=  str,
            help = 'path to gold-standards and file path file')

    parser.add_argument('-t','--last_n_records', type=int,
            help = 'Use last n records. defulat: 8',
            default =  8)

    args = parser.parse_args()
    
    
    opts = vars(args)
    run(**opts)


def run(gs_file_path, last_n_records):
    five_fold_cv(gs_file_path, last_n_records)


if __name__ == '__main__':
    main()
