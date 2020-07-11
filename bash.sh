#!/bin/bash
## $1 data base directory
perl split.pl ${1}
python train.py ${1}
python predict.py ${1}



