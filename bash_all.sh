#!/bin/bash

#example sh bash_all.sh ../../data
rm auc_all.txt
rm auprc_all.txt
rm auc_chunck_all.txt
rm auprc_chunck_all.txt
bash bash.sh 0 &
bash bash.sh 1 &
bash bash.sh 2 &
bash bash.sh 3 &
bash bash.sh 4 &

wait;
python evaluation.py 0
cat auc.txt >>auc_all.txt
cat auprc.txt >>auprc_all.txt
python evaluation.py 1
cat auc.txt >>auc_all.txt
cat auprc.txt >>auprc_all.txt
python evaluation.py 2
cat auc.txt >>auc_all.txt
cat auprc.txt >>auprc_all.txt
python evaluation.py 3
cat auc.txt >>auc_all.txt
cat auprc.txt >>auprc_all.txt
python evaluation.py 4
cat auc.txt >>auc_all.txt
cat auprc.txt >>auprc_all.txt
