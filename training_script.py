import torch
import warnings
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

# Suppress warnings and dataset progress bars
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()

# ==================== MAIN ====================
def main():
    # ===== CONFIG =====
    # Model and tokenizer settings
    MODEL_NAME = "google/flan-t5-base"  # Hugging Face model name; supports seq2seq tasks
    DATA_FILE = "data.json"             # Local JSON dataset file
    OUTPUT_DIR = "../ai_translater/model" # Directory to save model & tokenizer

    # Device and precision settings
    # If GPU is available, use mixed precision (bf16) and TF32 for performance
    # Otherwise, fallback to CPU with float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16           # Mixed precision reduces memory usage and speeds up training
        use_bf16, use_tf32 = True, True  # Enable bf16 and TensorFloat32
        torch.backends.cudnn.benchmark = True  # Auto-tune GPU kernels for better throughput
        print("[INFO] GPU detected: CUDA + bf16 + tf32")
    else:
        device = torch.device("cpu")
        dtype = torch.float32            # CPU only supports float32
        use_bf16, use_tf32 = False, False
        print("[INFO] No GPU detected: CPU + float32")
    print("[INFO] Using device:", device)

    # Training hyperparameters
    # These can be tuned based on dataset size and GPU memory
    TRAINING_CONFIG = {
        "per_device_train_batch_size": 16,  # Batch size per GPU/CPU core
        "gradient_accumulation_steps": 4,   # Accumulate gradients over multiple steps

        "num_train_epochs": 17,             # Total training epochs
        "learning_rate": 4e-4,              # AdamW learning rate
        "warmup_steps": 8,                  # Number of steps to gradually increase LR

        "optim": "adamw_torch_fused",       # Fused AdamW optimizer for speed on GPU
        "lr_scheduler_type": "linear",      # Linear LR decay

        "bf16": use_bf16,                   # Use bf16 if supported
        "tf32": use_tf32,                   # Use TF32 for faster matmul
        "fp16": False,                      # Disable fp16 (we're using bf16 instead)

        "save_strategy": "no",              # Disable checkpoint saving during training
        "eval_strategy": "no",              # Disable evaluation during training
        "logging_steps": 5,                 # Log every N steps

        "dataloader_num_workers": 0,        # Number of subprocesses for data loading
        "dataloader_pin_memory": False,     # Pin memory to GPU (can improve speed)
        "remove_unused_columns": False      # Keep all dataset columns (necessary for collator)
    }

    # ===== LOAD MODEL & TOKENIZER =====
    print(f"[INFO] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.config.use_cache = False  # Required when using gradient checkpointing
    model.to(device)
    print("[INFO] Model loading complete.")

    # ===== LOAD DATASET =====
    print("[INFO] Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")

    # ===== TOKENIZATION =====
    def preprocess(example):
        # Convert raw text to token IDs
        model_input = tokenizer(example["input"], max_length=256, truncation=True)
        labels = tokenizer(example["output"], max_length=256, truncation=True)["input_ids"]
        model_input["labels"] = torch.tensor(labels, dtype=torch.long)  # Labels must be tensor
        return model_input

    print("[INFO] Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, batched=False)
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
    tokenized_dataset.set_format(type="torch")
    print("[INFO] Tokenization complete.")

    # ===== DATA COLLATOR =====
    class FastDataCollator:
        """Pads input_ids, attention_mask, and labels dynamically per batch"""
        def __init__(self, tokenizer):
            self.pad_token_id = tokenizer.pad_token_id
            self.label_pad_token_id = -100  # HF ignores -100 when computing loss

        def __call__(self, features):
            input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True,
                                     padding_value=self.pad_token_id)
            attention_mask = pad_sequence([f.get("attention_mask", torch.ones_like(f["input_ids"]))
                                          for f in features], batch_first=True, padding_value=0)
            labels = pad_sequence([f["labels"] for f in features], batch_first=True,
                                  padding_value=self.label_pad_token_id)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    data_collator = FastDataCollator(tokenizer)

    # ===== TRAINING ARGUMENTS =====
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG
    )

    # ===== TRAINER =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ===== TRAIN =====
    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training complete.")

    # ===== SAVE MODEL =====
    print("[INFO] Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Model saved to {OUTPUT_DIR}")


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    main()
