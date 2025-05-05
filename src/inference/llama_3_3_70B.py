import os.path

# Keep this import order
from unsloth import FastLanguageModel
from transformers import GenerationConfig

from src.finetune.llama_3_3_70B import ALPACA_PROMPT, INSTRUCTION, DTYPE, LOAD_IN_4BIT, \
    OUTPUT_DIR, MAX_SEQ_LENGTH
from src.inference.models import LlamaSummarizer

MODEL_CONFIGS = {
    "elife_biobert_40_1e": {
        "output_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/generated_summaries/finetuned_length_40_epoch_1_{split}_elife_BioBERT_70B.json",
        "checkpoint_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/generated_summaries/finetuned_length_40_epoch_1_{split}_elife_BioBERT_70B.ckpt",
        "dataset_split": "extractive/length_40/{split}_elife_BioBERT.csv",
    },
    "elife_biobert_40_2e": {
        "output_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/generated_summaries/finetuned_length_40_epoch_2_{split}_elife_BioBERT_70B.json",
        "checkpoint_path": "/gscratch/stf/yongsinp/biolaysumm2025-task/generated_summaries/finetuned_length_40_epoch_2_{split}_elife_BioBERT_70B.ckpt",
        "dataset_split": "extractive/length_40/{split}_elife_BioBERT.csv",
    },
}


class UnslothLlamaSummarizer(LlamaSummarizer):
    def __init__(self, model_name, data_files: str, checkpoint_path: str, input_field: str = "extracted_summary"):
        self.model_name = model_name
        self.max_seq_length = MAX_SEQ_LENGTH
        self.max_new_tokens = 384
        self.dtype = DTYPE
        self.load_in_4bit = LOAD_IN_4BIT
        self.checkpoint_path = checkpoint_path
        self.checkpoint_rate = 10
        self.input_field = input_field
        self.dataset = self._load_dataset("suwmit", data_files)
        self._model, self._tokenizer = self._load_model()
        self._summaries = []

    def _load_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)

        model.generation_config = GenerationConfig(
            do_sample=False,
            # dola_layers="low",  # https://arxiv.org/abs/2309.03883
            max_new_tokens=self.max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        return model, tokenizer

    def generate(self, input_text, max_new_tokens=-1):
        self._model.generation_config.update(max_new_tokens = max_new_tokens if max_new_tokens != -1 else self.max_new_tokens)

        inputs = self._tokenizer(
            [
                ALPACA_PROMPT.format(
                    INSTRUCTION,  # instruction
                    input_text,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")
        input_length = inputs["input_ids"].shape[1]

        outputs = self._model.generate(**inputs, generation_config=self._model.generation_config, use_cache=True)
        return self._tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)

    def eval(self, max_new_tokens=-1):
        start_index = self._read_checkpoint()
        end_index = len(self.dataset)

        while start_index < end_index:
            input = self.dataset[self.input_field][start_index]
            output = self.generate(input, max_new_tokens)
            self._summaries.append(output[0].strip())

            if start_index % self.checkpoint_rate == 0:
                self._write_checkpoint(start_index)

            start_index += 1

        self._write_checkpoint(start_index)


model_name = os.path.join(OUTPUT_DIR, "final")
split = "validation"
config = MODEL_CONFIGS["elife_biobert_40_2e"]
checkpoint_path = config["checkpoint_path"].format(split=split)
data_files = config["dataset_split"].format(split=split)
summarizer = UnslothLlamaSummarizer(model_name, data_files=data_files, checkpoint_path=checkpoint_path)
result = summarizer.eval()
summarizer.save(config["output_path"].format(split=split))
