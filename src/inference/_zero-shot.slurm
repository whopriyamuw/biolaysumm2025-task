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
#SBATCH --chdir=/mmfs1/home/jcols/inference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jcols@uw.edu

eval "$(conda shell.bash hook)"
conda activate inference

python generate_zero_shot.py --output-path /gscratch/scrubbed/jcols/generated_summaries/length_20_validation_elife_BioBERT_preprocessed.jsonl --checkpoint-path /gscratch/scrubbed/jcols/generated_summaries/length_20_validation_elife_BioBERT_preprocessed.ckpt --dataset suwmit --dataset-split 20sent_preprocessed --batch-size 12 --input-field extracted_summary