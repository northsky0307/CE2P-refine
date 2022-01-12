#!/bin/bash
uname -a
#date
#env
date

DATA_DIRECTORY='./data/resize473/train'
DATA_LIST_PATH='./dataset/list/lip/trainList_rev_split.txt'
RESTORE_FROM='./dataset/MS_DeepLab_resnet_pretrained_init.pth'
SNAPSHOT_DIR='./snapshots/'
LR=7e-3
BATCHSIZE=55
STEPS=80000
GPU_IDS='0,1,2,3,4'
INPUT_SIZE='473,473'
NUM_CLASSES=20  
STARTITERS=0
 
python train.py --data-dir ${DATA_DIRECTORY} \
                          --data-list ${DATA_LIST_PATH} \
                          --input-size ${INPUT_SIZE} \
                          --num-classes ${NUM_CLASSES} \
                          --random-mirror \
                          --random-scale \
                          --gpu ${GPU_IDS} \
                          --learning-rate ${LR} \
                          --batch-size ${BATCHSIZE} \
                          --num-steps ${STEPS} \
                          --start-iters ${STARTITERS}\
                          --restore-from ${RESTORE_FROM} \
                          --snapshot-dir ${SNAPSHOT_DIR}
                          
python evaluate.py --restore-from snapshots/LIP_${STEPS}.pth --gpu 0
