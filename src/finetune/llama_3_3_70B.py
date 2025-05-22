import os.path

# Changing import order breaks things for some reason
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer

MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096  # Choose any! We auto support RoPE Scaling internally!
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

INSTRUCTION = "You are a specialist medical communicator responsible for translating biomedical articles into a clear, accurate 10-20 sentence summary for non-experts. The summary should be at a Flesch–Kincaid grade level of 10–14 and explain any technical terms."
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    inputs = examples["extracted_summary"]
    outputs = examples["summary"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ALPACA_PROMPT.format(INSTRUCTION, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


dataset = load_dataset("whopriyam2/SUWMIT-dataset", data_files="extractive/length_40/train_elife_BioBERT.csv",
                       split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, )

OUTPUT_DIR = "../../models/3.3_70B"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,  # Set this for 1 full training run.
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=50,
        report_to="none",  # Use this for WandB etc
    ),
)


def should_resume(directory_path):
    return any(os.scandir(directory_path))


resume = should_resume(OUTPUT_DIR)
trainer_stats = trainer.train(resume_from_checkpoint=resume)

# Save the final model after training
model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
