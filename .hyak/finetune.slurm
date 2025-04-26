#!/bin/bash

#SBATCH --job-name=ft_elife_extractive_20
#SBATCH --output=./logs/ft_elife_extractive_20_%j.out
#SBATCH --error=./logs/ft_elife_extractive_20_%j.err

#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

#SBATCH --export=all
#SBATCH --chdir=__PATH__  # Fill in
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=__USER_NAME__@uw.edu  # Fill in

eval "$(conda shell.bash hook)"
conda activate llama3
python finetune.py --data-files extractive/length_20/train_elife_BioBERT.csv --epochs 2 --data-split 0.25 --input extracted_summary --log-level DEBUG