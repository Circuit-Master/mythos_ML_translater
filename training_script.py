import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

# ===== CONFIG =====
MODEL_NAME = "google/flan-t5-base"
DATA_FILE = "data.json"
OUTPUT_DIR = "model"     # final fine-tuned model folder

print("[DEBUG] Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE)["train"]
print("[DEBUG] Dataset loaded.")
print(f"[DEBUG] Number of training samples: {len(dataset)}")

# ===== LOAD MODEL + TOKENIZER =====
print(f"[DEBUG] Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ===== TOKENIZATION FUNCTION =====
def preprocess(example):
    # T5 uses "input → label"
    model_input = tokenizer(
        example["input"],
        max_length=256,
        truncation=True
    )

    labels = tokenizer(
        example["output"],
        max_length=256,
        truncation=True
    )["input_ids"]

    model_input["labels"] = labels
    return model_input

print("[DEBUG] Tokenizing...")
tokenized_dataset = dataset.map(preprocess, batched=False)
print("[DEBUG] Tokenization complete.")

# ===== DATA COLLATOR =====
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ===== TRAINING CONFIG =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=5,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
)

# ===== TRAINER =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("[DEBUG] Starting training...")
trainer.train()
print("[DEBUG] Training complete.")

# ===== SAVE =====
print("[DEBUG] Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"[DEBUG] Model saved to {OUTPUT_DIR}")
