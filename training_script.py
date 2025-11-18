import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

def main():
    # ===== CONFIG =====
    MODEL_NAME = "google/flan-t5-base"
    DATA_FILE = "data.json"
    OUTPUT_DIR = "../hugging_face/model"

    print("[DEBUG] Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    print("[DEBUG] Dataset loaded.")
    print(f"[DEBUG] Number of training samples: {len(dataset)}")

    # ===== SELECT DEVICE =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] Using device:", device)
    torch.backends.cudnn.benchmark = True

    # ===== LOAD MODEL + TOKENIZER =====
    print(f"[DEBUG] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = model.to(device)

    # ===== TOKENIZATION FUNCTION =====
    def preprocess(example):
        model_input = tokenizer(
            example["input"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            example["output"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )["input_ids"]
        model_input["labels"] = labels
        return model_input

    print("[DEBUG] Tokenizing...")
    tokenized_dataset = dataset.map(preprocess, batched=False)
    print("[DEBUG] Tokenization complete.")

    # ===== DATA COLLATOR =====
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ===== TRAINING SETTINGS =====
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=40,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1500,
        bf16=True,
        fp16=False,
        save_total_limit=10,
        save_strategy="epoch",
        logging_steps=15,
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
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


if __name__ == "__main__":
    main()
