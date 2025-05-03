# SUWMIT @ BioLaySumm 2025

## Getting Started

This is a Python project bootstrapped with [`uv`](https://github.com/astral-sh/uv), an extremely
fast Python package and project manager, written in Rust.

### Installation

1. Clone the repository:
   ```sh
   git clone --recursive git@github.com:whopriyamuw/biolaysumm2025-task.git suwmit-biolaysumm
   cd suwmit-biolaysumm
   ```

2. Use a virtual environment:

   ```sh
   # With conda:
   conda create -n suwmit python=3.11.0
   conda active suwmit
   # With uv:
   uv venv 
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```sh
   # With pip:
   pip install .
   # With uv:
   uv sync
   ```

4. Download artifacts:

   ```sh
   python -m spacy download en_core_web_sm
   ```

> [!IMPORTANT]  
> Many scripts in this project rely on Linux x86_64 and NVIDIA GPUs. Before installing dependencies,
> please ensure that your computing environment is compatible.

## Usage

> [!NOTE]  
> The project's code is located in the `src` directory. To run scripts, you must use this directory
> as the working directory.

### Data scripts

- `download_datasets`. This script downloads the official `PLOS` and `eLife` datasets, storing all
  splits as CSV files in
  `data/raw/plos` and `data/raw/elife`, respectively.
   ```sh
   python -m data.download_datasets
   ```

- `explore_data`. This script extracts token statistics from a raw dataset and saves them to a CSV
  file in the
  `data/processed` directory. For a more detailed exploratory data analysis, refer to the Jupyter
  Notebook located at `notebooks/datasets_eda.ipynb`.
   ```sh
   python -m data.explore_data
   ```

- `normalize_metrics`. This script normalizes the evaluation metric scores using min-max
  normalization and calculates the
  average across three categories: relevance, factuality, and readability. It produces a sorted
  ranking of the evaluation results.
   ```sh
   python -m data.normalize_metrics
   ```

### Extractive summarization

The `extract/extractive_summariser.py` script generates extractive summaries using a specific
embedding model, dataset, and split configuration. These parameters are set globally within the
script's code. Once configured, you can run the following command:

   ```sh
   python -m extract.extractive_summariser
   ```

The extractive summaries will be saved in the `data/processed/extractive` directory.

### Fine-tuning

1. **Log in to Hyak**  
   Request access if you don’t have it already:  
   https://my.environment.uw.edu/esits/how-to-hyak/hyak-how-to-get-access/

2. **Create a user directory** (if you haven’t already):
   ```sh
   mkdir -p /gscratch/scrubbed/USERNAME
   ```

3. **Configure your environment** by adding to `~/.bashrc`:
   ```sh
   # Hugging Face  
   export HF_HOME="/gscratch/scrubbed/USERNAME/.cache/huggingface/hub"  
   export HF_TOKEN=YOUR_HF_TOKEN # Needed for downloading LLaMa 3 weights; request access on HuggingFace first
   ```

4. **Create a Conda configuration file**:
   ```sh
   echo "always_copy: true" >> ~/.condarc
   ```

5. **Install Miniconda**:
   ```sh
   salloc  
   cd /gscratch/scrubbed/USERNAME  
   mkdir miniconda3  
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
   chmod +x Miniconda3-latest-Linux-x86_64.sh  
   ./Miniconda3-latest-Linux-x86_64.sh -b -u -p miniconda3  
   rm Miniconda3-latest-Linux-x86_64.sh  
   source miniconda3/bin/activate  
   conda init --all
   ```

6. **Create and activate a new Conda environment**, then install dependencies:
   ```sh
   conda create -n llama3 python=3.11 -y  
   conda activate llama3  
   pip install torch torchtune torchao transformers datasets wandb --no-cache-dir
   ```

7. **Copy your project** under:
   ```sh
   /gscratch/scrubbed/USERNAME
   ```

8. **Change to the project directory**:
   ```sh
   cd /gscratch/scrubbed/USERNAME/project_directory/finetune
   ```

9. **Request a GPU and run your training job** (tested on NVIDIA A40/A100 with 64 GB RAM):
   ```sh
   eval "$(conda shell.bash hook)"  
   conda activate llama3  
   python finetune.py \
   --train-data extractive/length_40/train_elife_BioBERT.csv \
   --epochs 2 \
   --input extracted_summary
   ```

10. **Download LLaMa 3 weights**:
   ```sh
    tune download meta-llama/Meta-Llama-3.1-8B \
    --output-dir ./models/Llama-3.1-8B \
    --hf-token TOKEN
   ```

11. **Fine-tune LLaMa 3**:
   ```sh
    # Single-device:
    tune run lora_finetune_single_device --config CONFIG_NAME
    # Distributed (2 GPUs):
    tune run --nproc_per_node 2 lora_finetune_distributed --config CONFIG_NAME
   ```

### Inference

The inference module supports two execution modes: running the base LLaMa 3.1 8B Instruct model with
zero-shot prompting and executing a fine-tuned version of the same LLaMa model by loading its
adapter weights.

To run the **base model**, use the following command and specify the arguments detailed below.

   ```sh
    cd src/inference
    python generate_zero_shot.py 
   ```

- `--output-path`: Path to save the generated summaries. The script supports two formats: JSON lines
  and TXT.
- `--checkpoint-path`: Path to save the summaries checkpoint. This ensures the inference process
  remains resilient to pre-emption.
- `--dataset`: Name of the input dataset to use.
- `--dataset-split`: Dataset split or data_file to use.
- `--input-field`: Column name to use as input for the article text.
- `--batch-size`: Number of articles per batch.

To run a **fine-tuned model**, use the following command and specify the arguments detailed below.

   ```sh
    cd src/inference
    python generate_from_finetuned.py 
   ```

- `--model-name`: Name of the model configuration to use. Each configuration is specified within the
  script's code and includes the same parameters as the `generate_zero_shot.py` script, along with
  the adapter path, which indicates the directory of the fine-tuned LoRA weights.
- `--split`: Dataset split to use during inference (validation or test).

**SLURM jobs**

It is also possible to run inference jobs using SLURM
on [Hyak](https://hyak.uw.edu/docs/compute/scheduling-jobs/). This simplifies the process of
requesting a node with the appropriate hardware configuration needed to run the code. However, this
process requires configuring the project's virtual environment using `conda`, as outlined in step 5
of the fine-tuning instructions.

Example SLURM files are provided in `src/inference` and are structured internally as follows:

```shell
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
#SBATCH --chdir=/mmfs1/home/NETID/suwmit/src/inference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=NETID@uw.edu

eval "$(conda shell.bash hook)"
conda activate llama3

python generate_zero_shot.py --output-path OUTPUT_PATH --checkpoint-path CHECKPOINT_PATH --dataset suwmit --dataset-split "extractive/length_30/validation_elife_BioBERT.csv" --batch-size 8 --input-field extracted_summary
```

### Evaluation

Although this project includes several evaluation metrics under `src/metrics`, the preferred method
is to use the `evaluation` package at the root of this repository, which features a self-contained
implementation of the official evaluation pipeline for the BioLaySumm 2025 shared task.

This project can be configured by running the commands below. This will create a new `conda`
environment and install all the necessary dependencies.

```shell
cd src/evaluation
bash prepare_env.sh
```

To evaluate a specific output, use the following command:

```shell
python evaluation_final.py \
  --prediction_file predictions.json \
  --groundtruth_file references.json \
  --task_name "lay_summ" \
  > output.eval.txt
```

Where,

- `predictions.json` is a JSON Lines file where each line contains a JSON object with a
  `generated_caption` property that holds the predicted summary.
- `references.json` is a JSON Lines file where each line contains a JSON object with two properties:
  `document` and `reference`.

This evaluation pipeline also supports execution through SLURM jobs. An example file is available at `src/evaluation/run_evals.slurm`.

### Additional Resources

- [End-to-End Workflow with torchtune](https://pytorch.org/torchtune/0.6/tutorials/e2e_flow.html)
- [Disk Storage Management with Conda | Hyak](https://hyak.uw.edu/blog/conda-disk-storage/)
- [Python | Hyak](https://hyak.uw.edu/docs/tools/python/)
- [Globus | Hyak](https://hyak.uw.edu/docs/storage/globus/)

## Acknowledgments

Parts of this work were completed on Hyak, UW's high performance computing cluster. This resource
was funded by the UW student technology fee