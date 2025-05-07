"""
Script for fine-tuning LLaMa 3 Instruct models.
"""
import argparse
import copy
import logging
import os
import re
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime
from math import ceil
from pprint import pformat
from typing import Optional, Union

import torch
import yaml

from paths import *
from gpus import *


def update_config(original: dict, updates: dict) -> dict:
    """
    Updates the original config with the updates.
    This changes the original dictionary.

    Parameters:
        original: Original config dictionary.
        updates: Updates to be applied.
    Returns:
        dict: Updated config dictionary. This is the same instance as original.
    """
    for key, value in updates.items():
        if isinstance(value, Mapping) and key in original:
            original[key] = update_config(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class ModelTuner:
    _logger = None

    def __init__(self, model: str, qat: bool = False, hf_token: Optional[str] = None):
        """
        Initializes the ModelTuner class.

        Parameters:
            model: Model to fine-tune.
            hf_token: Hugging Face token for authentication. Searches environment variables if not provided.
        """
        self._initialize_class_attributes()
        self._model = model
        self._qat = qat
        self._hf_token = hf_token

    @property
    def gpu_count(self) -> int:
        """Number of GPUs available."""
        return self._gpu_count

    @property
    def gpu_model(self) -> str:
        """Model name of the GPU."""
        return self._gpu_model

    @property
    def model(self) -> str:
        """Model to fine-tune."""
        return self._model

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """
        Sets up and returns a logger with the specified name and level.

        Returns:
            logging.Logger: A logger instance.
        """
        logger = logging.getLogger("fine-tune")

        # Set handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @classmethod
    def _initialize_class_attributes(cls):
        if cls._logger is None:
            # Set up logger
            cls._logger = cls._setup_logger()
            cls.set_logger_level(logging.INFO)

            # Get device setup
            cls._gpu_count, cls._gpu_model = cls.get_gpus()

    @classmethod
    def set_logger_level(cls, level: Union[str, int] = logging.INFO) -> None:
        """
        Sets logging level for the class logger.

        Parameters:
            level: Logging level.
        """
        cls._logger.setLevel(level)
        for handler in cls._logger.handlers:
            handler.setLevel(level)

    @staticmethod
    def get_epochs_trained(output_dir: str) -> int:
        """Checks the maximum number of epochs the model has been trained before."""
        regex_epoch = re.compile(r'^epoch_(\d+)$')
        epochs = [0]

        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if match := regex_epoch.match(file):
                    epochs.append(int(match.group(1)) + 1)

        return max(epochs)

    @staticmethod
    def is_epoch_completed(output_dir: str, epoch: int) -> bool:
        """
        Checks if the specified epoch has been completed.

        Parameters:
            output_dir: Directory where the model is saved.
            epoch: Epoch number to check.

        Returns:
            bool: True if the epoch has been completed, False otherwise.
        """
        return os.path.exists(os.path.join(output_dir, f"epoch_{epoch}", "original"))

    @classmethod
    def get_gpus(cls) -> tuple[int, str]:
        """
        Returns available GPUs.
        Requires PyTorch and NVIDIA GPUs.

        Returns:
            (gpu_count, gpu_model): A tuple containing the number of GPUs and their model name.
        """
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_model = re.search(rf"{'|'.join(GPUS.keys())}", gpu_name)

            if not gpu_model:
                cls._logger.warning(
                    f"\nUnrecognized GPU: {gpu_name}.\n"
                    f"Supported GPUs: {', '.join(GPUS.keys())}.\n"
                    f"Using default settings."
                )

            return torch.cuda.device_count(), gpu_model.group(0) if gpu_model else gpu_name
        else:
            raise RuntimeError("No GPU found.")

    @classmethod
    def download_model(cls, model: str, hf_token: Optional[str] = None) -> None:
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
            cls._logger.info(f"Downloading {model} model to: {model_path}")
            cls._logger.info("This may take a while.")

            cmd = [
                "tune", "download",
                model,
                "--hf-token", hf_token,
                "--output-dir", model_path,
                "--ignore-patterns", "original/consolidated.00.pth",
            ]
            subprocess.run(cmd, check=True)

    def create_config(self, model: str, output_dir: str, source: str, data_files: str, split_begin: int, split_end: int,
                      epochs: int = 1, batch_size: int = 1, input: Optional[str] = None, output: Optional[str] = None,
                      resume: bool = False) -> str:
        """
        Creates a new config file for the fine-tuning process.
        Parameters:
            ...
        Returns:
            str: Path to the new config file.
        """
        # Load the default config file
        config_file = "{}{}.yaml"
        default_config_file = config_file.format(model, "_qat" if self._qat else "")
        default_config_path = os.path.join(CONFIG_DIR, "torchtune_org", default_config_file)
        with open(default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)

        # Create changes to the config
        modifications = {
            'checkpoint_dir': os.path.join(MODEL_DIR, self._model.split("/")[-1]),
            'output_dir': output_dir,
            'dataset': {
                'source': source,
                'data_files': data_files,
                'column_map': {
                    'input': input or 'article',
                    'output': output or 'summary'
                },
                'split': f"train[{split_begin}%:{split_end}%]",
            },
            'resume_from_checkpoint': resume,
            'should_load_recipe_state': resume,
            'epochs': epochs,
            'batch_size': batch_size,
        }

        # Update the default config with the modifications
        new_config = update_config(copy.deepcopy(default_config), modifications)

        self._logger.debug(f"\n##### Default Config #####\n"
                          f"{pformat(default_config)}\n\n")
        self._logger.debug(f"\n##### Changes #####\n"
                          f"{pformat(modifications)}\n\n")
        self._logger.debug(f"\n##### New Config #####\n"
                          f"{pformat(new_config)}\n\n")

        # Save the new config to a file
        new_config_dir = os.path.join(CONFIG_DIR, "torchtune_run")
        new_config_file = config_file.format(model,
                                             ("_qat" if self._qat else "")
                                             + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        new_config_path = os.path.join(new_config_dir, new_config_file)
        os.makedirs(new_config_dir, exist_ok=True)
        with open(new_config_path, 'w') as file:
            yaml.dump(new_config, file, default_flow_style=False)

        return new_config_path

    @classmethod
    def _run_torchtune(cls, config_path: str) -> subprocess.Popen:
        cmd = (["tune", "run"] +
               (["--nproc_per_node", str(cls._gpu_count)] if cls._gpu_count > 1 else []) +
               (["lora_finetune_distributed"] if cls._gpu_count > 1 else ["lora_finetune_single_device"]) +
               ["--config", config_path])

        cls._logger.info(f"Running command: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    def finetune(self, source: str, data_files: str, epochs: int = 1, batch_size: int = -1, data_split: float = -1,
                 input: Optional[str] = None, output: Optional[str] = None) -> None:
        """
        Fine-tunes the model with the specified parameters.

        Parameters:
            source: Path or name of the dataset (e.g., Hugging Face dataset name).
            data_files: Path(s) to source data file(s).
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
            data_split: Data split for training.
            input: Input column for the model.
            output: Target output column for the model.
        """
        self.download_model(self._model, self._hf_token)

        # Output directory
        format_name = lambda s: s.translate(str.maketrans("/\\-", "___"))
        dataset = format_name(source) + (f"_{format_name(data_files).rsplit('.', maxsplit=1)[0]}" if data_files else "")
        model = self._model.split('/')[-1].lower().replace("-", "_")
        output_dir = os.path.join(MODEL_DIR, f"finetuned_{dataset}_{model}")

        # Todo: Halve batch size for the 70B model
        # Batch size
        batch_size = GPUS[self._gpu_model].batch_size if batch_size == -1 else batch_size

        # Adjust epochs based on data_split
        data_split = GPUS[self._gpu_model].data_split * self._gpu_count if data_split == -1 else data_split
        split_epochs = ceil(1 / data_split)  # Each split becomes an epoch
        epochs_needed = max(split_epochs * epochs, epochs)
        epochs_trained = self.get_epochs_trained(output_dir)

        self._logger.debug(f"\n"
                           f"{self.gpu_count = }\n"
                           f"{self.gpu_model = }\n"
                           f"{batch_size = }\n"
                           f"{data_split = }\n"
                           f"{epochs_needed = }\n"
                           f"{epochs_trained = }\n")

        while (epochs_trained := self.get_epochs_trained(output_dir)) < epochs_needed:
            self._logger.info(f"\nEpoch: {epochs_trained}/{epochs_needed}")

            # Calculate dataset split
            split_begin = int(epochs_trained % split_epochs * data_split * 100)
            split_end = int(min(split_begin + data_split * 100, 100))

            # Create a new config for fine-tuning
            new_config_path = self.create_config(model, output_dir, source, data_files, split_begin, split_end, epochs_needed,
                                                 batch_size, input, output, resume=bool(epochs_trained))

            # Fine-tune the model
            process = self._run_torchtune(new_config_path)

            # Wait for the epoch to finish
            while not self.is_epoch_completed(output_dir, epochs_trained):
                time.sleep(30)

            self._logger.info("Terminating fine-tuning process after 1 epoch.")
            time.sleep(5)
            process.terminate()

        self._logger.info(f"Model already trained for {epochs_trained}/{epochs_needed} epochs. Exiting...")


def main():
    parser = argparse.ArgumentParser(description="Script for fine-tuning LLaMa 3 Instruct models.")

    # Dataset arguments
    parser.add_argument('--source', type=str, default='whopriyam2/SUWMIT-dataset',
                        help='Path or name of the dataset (default: whopriyam2/SUWMIT-dataset')
    parser.add_argument('--data-files', type=str, default=None,
                        help='Path(s) to source data file(s) (default: None)')
    parser.add_argument('--input', type=str, default='article',
                        help='Input column for the model (default: article)')
    parser.add_argument('--output', type=str, default='summary',
                        help='Target output column for the model (default: summary)')

    # Training arguments
    parser.add_argument('--model', type=str, default='3.1-8B',
                        choices=['3.1-8B', '3.3-70B'],
                        help='Name of the model to train (default: 3.1-8B)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Epochs to train the model (default: 1)')
    parser.add_argument('--data-split', type=float, default=-1,
                        help='Data split for training (default: -1, automatically calculated based on GPU model and number)')
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='Batch size for training (default: -1, automatically set based on GPU model)')
    parser.add_argument('--qat', action='store_true',
                        help='Enable quantization-aware training (default: False)')

    # Logging arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level (default: INFO)')

    args = parser.parse_args()

    if args.qat and args.model == '3.1-8B':
        raise ValueError("Quantization-aware training is only supported for the 3.3-70B model.")

    model = "meta-llama/Llama-{}-Instruct".format(args.model)
    tuner = ModelTuner(model, args.qat)
    tuner.set_logger_level(args.log_level)
    tuner.finetune(args.source, args.data_files, args.epochs, args.batch_size, args.data_split, args.input, args.output)


if __name__ == "__main__":
    main()
