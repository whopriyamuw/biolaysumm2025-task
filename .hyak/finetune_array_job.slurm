#!/bin/bash

#SBATCH --job-name=ft_elife_extractive_8B_array
#SBATCH --output=./logs/%x_%A_%a.out

#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=192G
#SBATCH --time=24:00:00

#SBATCH --array=0-7

#SBATCH --export=all
#SBATCH --chdir=/gscratch/stf/yongsinp/biolaysumm2025-task/src/finetune
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yongsinp@uw.edu

LEN_LIST=(
    10
    20
    30
    40
)

LEN=${LEN_LIST[${SLURM_ARRAY_TASK_ID}%4]}

FILE_LIST=(
    "new_files/train_elife_BioBERT_${LEN}sent_concat_abstract.csv"
    "new_files/train_elife_BioBERT_${LEN}sent_exclude_abstract.csv"
)

FILE=${FILE_LIST[${SLURM_ARRAY_TASK_ID}/4]}

echo "Training on: ${FILE}"

eval "$(conda shell.bash hook)"
conda activate llama3
python finetune.py \
--source whopriyam2/SUWMIT-dataset \
--train-data ${FILE} \
--model 3.1-8B \
--epochs 2 \
--batch-size 2 \
--data-split 0.25 \
--input extracted_summary \
--log-level INFO