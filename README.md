# Sinhala Essay Scoring Model - Training Pipeline

This repository contains the complete training pipeline for the Bias-Aware Sinhala Essay Scoring Engine. It handles dataset preparation, model definition (Multi-Head XLM-Roberta), and training execution.

## 🚀 Overview

The model is fine-tuned on a specialized dataset containing both:
1.  **General Sinhala Essays:** 130+ essays from standard curriculum.
2.  **Dyslexic Essays:** 160+ essays (real and synthetic) exhibiting dyslexic patterns (spelling errors, letter reversals).

**Base Model:** `xlm-roberta-large`
**Architecture:** Multi-Head Regressor (4 heads: Richness, Organization, Technical, Total)

## 📂 Project Structure

```bash
scoring-model-training/
├── training/
│   ├── dataset_loader.py       # Loads & cleans CSV, handles input renaming
│   ├── model_multitask_xlmr.py # Defines the Multi-Head XLM-R architecture
│   ├── train_model.py          # Main training loop (PyTorch + Transformers)
│   └── eval_model.py           # Evaluation scripts
├── generate_dyslexic_essays.py # Script for synthetic data augmentation
├── merge_dyslexic_datasets.py  # Utility to combine real & synthetic data
├── sinhala_dataset_final_with_dyslexic.csv # The FINAL training dataset
└── requirements.txt            # Python dependencies
```

## 🛠️ Setup & Installation

1.  **Create venv:**
    ```bash
    python -m venv venv311
    .\venv311\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Training Workflow

### 1. Data Preparation
The dataset `sinhala_dataset_final_with_dyslexic.csv` is already prepared. It contains:
- `input_text`: The essay content.
- `grade`: Student grade (3-8).
- `dyslexic_flag`: Boolean indicating if the essay has dyslexic patterns.
- `richness_5`, `organization_6`, `technical_3`, `total_14`: Teacher scores.

### 2. Run Training
To retrain the model from scratch:

```bash
python -m training.train_model
```

**Configuration (inside `train_model.py`):**
- **Epochs:** 20
- **Batch Size:** 4
- **Learning Rate:** 1e-5 (Lowered for stability)
- **Gradient Clipping:** 1.0

### 3. Model Output
The trained model is saved to:
```bash
./xlm-roberta-large-sinhala-multihead/
```
This folder contains:
- `config.json`
- `model.safetensors`
- `tokenizer.json`

## 📊 Evaluation & Bias Detection

After training, the model is moved to the **Backend Repository** (`bias-aware-scoring-engine`) for deployment and fairness evaluation using the `firestore_fairness_eval.py` script.

## 🔄 Recent Updates
- **Merged Dataset:** Combined real and synthetic dyslexic essays for balanced training.
- **Fixed NaN Loss:** Implemented strict data cleaning and gradient clipping.
- **Grade-Aware:** Model accepts `grade` as an additional input feature.

---
*Maintained by the Akura Research Team.*
