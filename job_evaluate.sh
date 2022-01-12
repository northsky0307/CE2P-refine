#!/bin/bash
uname -a
#date
#env
date

DATA_DIRECTORY='./dataset/data'
DATA_LIST_PATH='./dataset/list/lip/valList.txt' 
NUM_CLASSES=20 
RESTORE_FROM='./snapshots_val/LIP_edge_115000.pth'
SAVE_DIR='./outputs_val/' 
INPUT_SIZE='473,473'
GPU_ID=0
 
python evaluate.py --data-dir ${DATA_DIRECTORY} \
                   --data-list ${DATA_LIST_PATH} \
                   --input-size ${INPUT_SIZE} \
                   --num-classes ${NUM_CLASSES} \
                   --save-dir ${SAVE_DIR} \
                   --gpu ${GPU_ID} \
                   --restore-from ${RESTORE_FROM}
                           
