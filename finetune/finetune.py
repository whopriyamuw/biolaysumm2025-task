"""
Script for fine-tuning LLaMa 3.1 8B Instruct model.
"""
import argparse
import collections
import inspect
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pprint import pformat
from typing import Optional

import yaml
from blobfile import makedirs

from paths import MODEL, MODEL_DIR, CONFIG_FILE, CONFIG_DIR

GPU_SPEED = {'A40': 0.25, 'A100': 1}  # Epochs per 8 hours


def setup_logger(name: str = "finetune", level: str = logging.DEBUG) -> logging.Logger:
    """
    Sets up and returns a logger with the specified name and level.

    Parameters:
        name: Name of the logger.
        level: Logging level.

    Returns:
        logging.Logger: A logger instance.
    """
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


def download_model(model: str = MODEL, hf_token: Optional[str] = None) -> None:
    """
    Downloads the model from Hugging Face.

    Parameters:
        model: Model to download from Hugging Face.
        hf_token: Hugging Face token for authentication. Searches environment variables if not provided.
    """
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN', None)

    model_path = os.path.join(MODEL_DIR, model.split("/")[-1])

    # Download the model if it doesn't exist.
    if not os.path.exists(model_path):
        logger.info(f"Downloading {model} model to: {model_path}")
        logger.info(f"This may take a while.")

        cmd = [
            "tune", "download",
            model,
            "--hf-token", hf_token,
            "--output-dir", model_path,
            "--ignore-patterns", "original/consolidated.00.pth",
        ]
        subprocess.run(cmd, check=True)


def finetune(data_files: str, epochs: int = 1):
    dataset_name = data_files.replace("/", "_").replace("\\", "_").rsplit(".", maxsplit=1)[0]
    output_dir = os.path.join(MODEL_DIR, f"finetuned_{dataset_name}")
    log_config(output_dir, dataset_name)

    # Get device setup
    gpu_count, gpu_model = get_gpus()
    gpu_speed = GPU_SPEED.get(gpu_model, 1) * gpu_count
    device_setup = "single" if gpu_count == 1 else "distributed"
    log_config(gpu_count, gpu_model, gpu_speed, device_setup)

    # Calculate epochs
    epochs_needed = int(max(epochs / gpu_speed, epochs))  # Adjust epochs based on the GPU model
    epochs_trained = get_epochs_trained(output_dir)
    log_config(epochs, epochs_needed, epochs_trained)

    if epochs_needed <= epochs_trained:
        logger.info(f"Model already trained for {epochs_trained}/{epochs_needed} epochs. Exiting...")
        sys.exit()

    # Calculate dataset split
    split_begin, split_end = mod(gpu_speed * epochs_trained * 100), mod(gpu_speed * (epochs_trained + 1) * 100, True)
    log_config(split_begin, split_end)

    # Create config and fine-tune
    new_config_path = create_config(output_dir, data_files, split_begin, split_end, epochs_needed, device_setup,
                                    resume=bool(epochs_trained))

    # Fine-tune the model
    _finetune(new_config_path, gpu_count)


def log_config(*args, **kwargs) -> None:
    frame = inspect.currentframe()
    try:
        # Get the caller's local variables
        arg_names = frame.f_back.f_locals
        # Log positional arguments with their names
        if args:
            # Todo: Fix where args have same values and get printed again
            logger.debug("\n" + "\n".join(f"{name}: {value}" for name, value in arg_names.items() if value in args))
        # Log keyword arguments
        if kwargs:
            logger.debug("\n" + "\n".join(f"{key}: {value}" for key, value in kwargs.items()))
    finally:
        del frame


def mod(x: int, end: bool = False) -> int:
    """
    Returns the modulus of x with respect to 100.

    Parameters:
        x: The value to be modded.
        end: If True, returns 100 if x % 100 is 0.

    Returns:
        int: The modulus of x with respect to 100.
    """
    if end and x % 100 == 0:
        return 100
    return int(x % 100)


def get_gpus() -> tuple[int, str]:
    """
    Returns available GPUs.
    Requires PyTorch and NVIDIA GPUs.

    Returns:
        (gpu_count, gpu_model): A tuple containing the number of GPUs and their model name.
    """
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_model = re.search(rf"{'|'.join(GPU_SPEED.keys())}", gpu_name)

        if not gpu_model:
            logger.Error(f"Currently supported GPUs: {', '.join(GPU_SPEED.keys())}")
            raise ValueError(f"Unrecognized GPU: {gpu_name}")

        return torch.cuda.device_count(), gpu_model.group(0)
    else:
        raise RuntimeError("No GPU found.")


def get_epochs_trained(output_dir: str) -> int:
    """Checks the maximum number of epochs the model has been trained before."""
    regex_epoch = re.compile(r'^epoch_(\d+)$')
    epochs = [0]

    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if match := regex_epoch.match(file):
                epochs.append(int(match.group(1)) + 1)

    return max(epochs)


def create_config(output_dir: str, data_files: str, split_begin: int, split_end: int,
                  epochs: int = 1, device_setup: str = "single", resume: bool = False) -> str:
    """
    Creates a new config file for the fine-tuning process.
    Parameters:
        ...
    Returns:
        str: Path to the new config file.
    """
    # Load the default config file
    default_config_file = CONFIG_FILE.format(device_setup)
    default_config_path = os.path.join(CONFIG_DIR, "torchtune_org", default_config_file)
    with open(default_config_path, 'r') as f:
        default_config = yaml.safe_load(f)

    # Create changes to the config
    modifications = {
        'checkpoint_dir': os.path.join(MODEL_DIR, MODEL.split("/")[-1]),
        'output_dir': output_dir,
        'dataset': {
            'data_files': data_files,
            'split': f"train[{split_begin}%:{split_end}%]",
        },
        'resume_from_checkpoint': resume,
        'should_load_recipe_state': resume,
        'epochs': epochs,
    }

    # Update the default config with the modifications
    new_config = update_config(default_config, modifications)

    logger.debug(f"\n##### Default Config #####\n"
                 f"{pformat(default_config)}\n\n")
    logger.debug(f"\n##### Changes #####\n"
                 f"{pformat(modifications)}\n\n")
    logger.debug(f"\n##### New Config #####\n"
                 f"{pformat(new_config)}\n\n")

    new_config_dir = os.path.join(CONFIG_DIR, "torchtune_run")
    new_config_file = CONFIG_FILE.format(f"{device_setup}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    new_config_path = os.path.join(new_config_dir, new_config_file)
    makedirs(new_config_dir)
    with open(new_config_path, 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)

    return new_config_path


def _finetune(config_path: str, gpu_count: int = 1) -> None:
    cmd = (["tune", "run"] +
           (["--nproc_per_node", str(gpu_count)] if gpu_count > 1 else []) +
           (["lora_finetune_distributed"] if gpu_count > 1 else ["lora_finetune_single_device"]) +
           ["--config", config_path])

    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def update_config(original, updates):
    for key, value in updates.items():
        if isinstance(value, collections.abc.Mapping) and key in original:
            original[key] = update_config(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def main():
    parser = argparse.ArgumentParser(description="Script for fine-tuning LLaMa 3.1 8B Instruct model.")

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset for fine-tuning.')

    # Optional arguments
    parser.add_argument('--epochs', type=int, default=1,
                        help='Epochs to train the model (default: 1)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level (default: INFO)')

    args = parser.parse_args()

    # Set logger level
    logger.setLevel(args.log_level)

    download_model()
    finetune(args.dataset, args.epochs)


if __name__ == "__main__":
    main()
