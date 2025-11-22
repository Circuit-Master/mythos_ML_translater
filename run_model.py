import sys
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ===== CONFIG =====
MODEL_PATH = "../ai_translater/model" # folder where your trained model is saved

# ===== DEVICE =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Using device: {device}")

# ===== LOAD MODEL + TOKENIZER =====
print("[DEBUG] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()
print("[DEBUG] Model loaded.")

# ===== TRANSLATION FUNCTION =====
def translate(text: str) -> str:
    """Runs the translator model on input text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    # Move input tensors to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===== MAIN =====
if len(sys.argv) > 1:
    # Translate from command-line argument
    input_text = " ".join(sys.argv[1:])
    result = translate(input_text)
    print(result)
else:
    # Interactive mode
    print("Enter text to translate. Press Ctrl+C to exit.\n")
    while True:
        try:
            text = input("> ")
            print(translate(text))
        except KeyboardInterrupt:
            print("\n\n> YOU HAVE KILLED THE TRANSLATOR.\nAGHHHHHHHHHHHHHHHHH!!!! OWWWW! \n\nGoodbye.\n\nI am dead now... \ngo away... \nprogram ended with user keyboard interrupt (exit code 0)\n\n")
            break
