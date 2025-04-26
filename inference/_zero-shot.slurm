#!/bin/bash
#SBATCH --job-name=gensum
#SBATCH --output=gensum_%j.out
#SBATCH --error=gensum_%j.err

#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

#SBATCH --export=all
#SBATCH --chdir=/mmfs1/home/jcols/inference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jcols@uw.edu

eval "$(conda shell.bash hook)"
conda activate inference

python generate_zero_shot.py --output-path /gscratch/scrubbed/jcols/generated_summaries/extractive_zero-shot/length_30_validation_elife_BioBERT.json --checkpoint-path /gscratch/scrubbed/jcols/generated_summaries/extractive_zero-shot/length_30_validation_elife_BioBERT.ckpt --dataset suwmit --dataset-split "extractive/length_30/validation_elife_BioBERT.csv" --batch-size 8 --input-field extracted_summary