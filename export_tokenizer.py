from transformers import AutoTokenizer

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Save tokenizer into your model output folder
tokenizer.save_pretrained("./xlm-roberta-large-sinhala-multihead")
