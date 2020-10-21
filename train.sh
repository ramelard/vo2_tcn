#!/bin/bash

source ~/tensorflow/bin/activate
cd src/
python main.py --epochs 100 --seq_step_train 1 --feature_list "WR,HR,VE,BF,HRR" --nb_filters $1 --kernel_size $2 --max_dilation_pow $3 --dropout_rate $4 --lr $5 --note $6 --log_dir ~/scratch/logs/$7/ --chkpt_dir ~/scratch/chkpts/$7
deactivate
