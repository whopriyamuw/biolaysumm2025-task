import os
import subprocess
from dataclasses import asdict

import torch
from datasets import Split
from llama_cookbook.configs import train_config as TrainConfig
from llama_cookbook.configs import lora_config as LORA_CONFIG
from llama_cookbook.utils import train, evaluation
from llama_cookbook.utils.dataset_utils import get_dataloader
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch import optim
from torch.optim.lr_scheduler import StepLR
from transformers import BitsAndBytesConfig
from transformers import LlamaForCausalLM, AutoTokenizer

from config.data import BioLaySummConfig, Dataset


def download_biolaysumm_datasets() -> None:
    """Checks for the datasets and downloads them if not present."""
    download_script_path = os.path.abspath('../download_datasets.py')
    script_dir = os.path.dirname(download_script_path)
    dataset_dir = os.path.join(script_dir, "biolaysumm_dataset")

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' does not exist. Downloading datasets...")

        try:
            subprocess.run(['python', download_script_path], check=True, cwd=script_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error while downloading data: {e}")
    else:
        print(f"Dataset directory '{dataset_dir}' already exists. Skipping download.")


def finetune(train_config, dataset_config):
    """Fine-tunes the model on the BioLaySumm dataset."""
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = LlamaForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        quantization_config=quantization_config,
        use_cache=False,
        attn_implementation="sdpa" if config.use_fast_kernels else None,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader = get_dataloader(tokenizer, dataset_config, train_config, Split.TRAIN)
    eval_dataloader = get_dataloader(tokenizer, dataset_config, train_config, Split.VALIDATION)

    lora_config = LORA_CONFIG()
    lora_config.r = 8
    lora_config.lora_alpha = 32
    lora_config.lora_dropout: float = 0.01

    peft_config = LoraConfig(**asdict(lora_config))

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start fine-tuning
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run=None,
    )

    # Save fine-tuned model
    model.save_pretrained(train_config.output_dir)

    # Evaluate fine-tuned model
    evaluation(model, train_config, eval_dataloader, None, tokenizer, None)


if __name__ == '__main__':
    # Todo: Replace print with logging
    train_config = TrainConfig()
    train_config.model_name = "meta-llama/Meta-Llama-3.1-8B"
    train_config.num_epochs = 1
    train_config.run_validation = False
    train_config.gradient_accumulation_steps = 4
    train_config.batch_size_training = 4
    train_config.lr = 3e-4
    train_config.use_fast_kernels = True
    train_config.use_fp16 = True
    train_config.context_length = 4096
    train_config.batching_strategy = "packing"
    train_config.output_dir = "./models/llama-3.1-8B-biolaysumm"
    train_config.use_peft = True

    data_config = BioLaySummConfig
    data_config.dataset = Dataset.elife.value

    finetune(train_config, data_config)
