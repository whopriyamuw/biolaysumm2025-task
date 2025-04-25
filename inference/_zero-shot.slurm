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

python generate_zero_shot.py --output-path /gscratch/scrubbed/jcols/generated_summaries/validation_elife_llama-zero-shot.json --checkpoint-path /gscratch/scrubbed/jcols/inference/validation_elife_llama-zero-shot.ckpt --dataset elife --dataset-split validation