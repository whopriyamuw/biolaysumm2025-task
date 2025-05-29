#!/bin/bash
#SBATCH --job-name=gensum
#SBATCH --output=gensum_%j.out
#SBATCH --error=gensum_%j.err

#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

#SBATCH --export=all
#SBATCH --chdir=/mmfs1/home/jcols/biolaysumm/src/inference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jcols@uw.edu

eval "$(conda shell.bash hook)"
conda activate inference

python generate_zero_shot.py \
    --output-path /gscratch/stf/jcols/biolaysum_summaries/elife_validation_extractive_30_preprocessed.jsonl \
    --checkpoint-path /gscratch/stf/jcols/biolaysum_summaries/elife_validation_extractive_30_preprocessed.ckpt \
    --dataset suwmit \
    --dataset-split preprocessed/elife_validation_extractive_30_preprocessed.csv \
    --batch-size 12 \
    --input-field transformed