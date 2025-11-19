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
    TrainingArguments,
    TrainerCallback
)

# Suppress Hugging Face FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # ===== CONFIG =====
    MODEL_NAME = "google/flan-t5-base"
    DATA_FILE = "data.json"
    OUTPUT_DIR = "../hugging_face/model"

    # ===== DEVICE & DTYPE =====
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Use bf16 on modern NVIDIA GPUs if supported
        dtype = torch.bfloat16
        use_bf16 = True
        use_tf32 = True

        print("[INFO] GPU detected: Using CUDA + bf16 + tf32")
        torch.backends.cudnn.benchmark = True  # Optimize GPU kernels


    else:
        device = torch.device("cpu")

        # CPU cannot use bf16 or tf32
        dtype = torch.float32
        use_bf16 = False
        use_tf32 = False

        print("[INFO] No GPU detected: Attempting CPU + float32")

    print("[INFO] Using device:", device)

    # ===== LOAD MODEL + TOKENIZER =====
    print(f"[INFO] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )
    model.config.use_cache = False
    model.to(device)
    print(f"[INFO] Model Loaded.")

    # ===== LOAD DATASET =====
    print("[INFO] Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")


    # ===== TOKENIZATION FUNCTION =====
    def preprocess(example):
        # Tokenize input
        model_input = tokenizer(example["input"], max_length=256, truncation=True)
        # Tokenize output/labels
        labels = tokenizer(example["output"], max_length=256, truncation=True)["input_ids"]
        # Convert to torch tensor (labels will be padded in collator)
        model_input["labels"] = torch.tensor(labels, dtype=torch.long)
        return model_input

    print("[INFO] Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, batched=False)
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
    tokenized_dataset.set_format(type="torch")
    print("[INFO] Tokenization complete.")

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

        warmup_steps=8,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,

        learning_rate=4e-4,
        num_train_epochs=15,

        bf16=use_bf16,
        tf32=use_tf32,
        fp16=False,   

        optim="adamw_torch_fused",
        lr_scheduler_type="linear",

        save_strategy="no",
        eval_strategy="no",

        logging_steps=5,

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
    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training complete.")

    # ===== SAVE MODEL =====
    print("[INFO] Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

