import os
import torch
import warnings
import numpy as np
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)

# Suppress Hugging Face FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # ===== CONFIG =====
    MODEL_NAME = "google/flan-t5-base"
    DATA_FILE = "data.json"
    OUTPUT_DIR = "../hugging_face/model"

    # ===== LOAD DATASET =====
    print("[DEBUG] Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    print(f"[DEBUG] Dataset loaded: {len(dataset)} samples")

    # ===== DEVICE =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] Using device:", device)
    torch.backends.cudnn.benchmark = True  # Optimize GPU kernels

    # ===== LOAD MODEL + TOKENIZER =====
    print(f"[DEBUG] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16  # Use bf16 for faster GPU training
    )
    model.config.use_cache = False
    model.to(device)

    # ===== TOKENIZATION FUNCTION =====
    def preprocess(example):
        # Tokenize input
        model_input = tokenizer(example["input"], max_length=256, truncation=True)
        # Tokenize output/labels
        labels = tokenizer(example["output"], max_length=256, truncation=True)["input_ids"]
        # Convert to torch tensor (labels will be padded in collator)
        model_input["labels"] = torch.tensor(labels, dtype=torch.long)
        return model_input

    print("[DEBUG] Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, batched=False)
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
    tokenized_dataset.set_format(type="torch")
    print("[DEBUG] Tokenization complete.")

    # ===== FAST DATA COLLATOR =====
    class FastDataCollator:
        """Pads variable-length sequences in a batch to the same length for fast stacking."""
        def __init__(self, tokenizer):
            self.pad_token_id = tokenizer.pad_token_id
            self.label_pad_token_id = -100  # HF convention for ignored positions

        def __call__(self, features):
            # Pad input_ids
            input_ids = pad_sequence(
                [f["input_ids"] for f in features],
                batch_first=True,
                padding_value=self.pad_token_id
            )
            # Pad attention_mask (1 where input exists, 0 where padded)
            attention_mask = pad_sequence(
                [f["attention_mask"] if "attention_mask" in f else torch.ones_like(f["input_ids"]) for f in features],
                batch_first=True,
                padding_value=0
            )
            # Pad labels
            labels = pad_sequence(
                [f["labels"] for f in features],
                batch_first=True,
                padding_value=self.label_pad_token_id
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

    data_collator = FastDataCollator(tokenizer)

    # ===== TRAINING CONFIG =====
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        warmup_steps=15,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=5,

        learning_rate=3e-4,
        num_train_epochs=10,

        bf16=True,
        fp16=False,
        tf32=True,

        optim="adamw_torch_fused",
        lr_scheduler_type="linear",

        save_strategy="no",
        eval_strategy="no",

        logging_steps=10,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        remove_unused_columns=False,
    )

    # ===== TRAINER =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ===== START TRAINING =====
    print("[DEBUG] Starting training...")
    trainer.train()
    print("[DEBUG] Training complete.")

    # ===== SAVE MODEL =====
    print("[DEBUG] Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[DEBUG] Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
