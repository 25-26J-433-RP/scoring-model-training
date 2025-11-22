This repository contains the training pipeline for the Sinhala Essay Scoring Engine.
It prepares datasets, defines the multi-head XLM-R model, and will train the final scoring model once teacher rubric scores arrive.

Status

This repo does not train a model yet because teacher labels are not available.
It includes a complete Phase 6 training skeleton, fully ready to activate when labels arrive.

Structure
training/
  dataset_loader.py
  model_multitask_xlmr.py
  train_model.py
  eval_model.py

prepare_sinhala_dataset_v2.py
requirements.txt
readme.md

Workflow (When Teacher Scores Arrive)

Place the labelled CSV in this folder

Run dataset prep:

python prepare_sinhala_dataset_v2.py


Train the model:

python -m training.train_model


Evaluate:

python -m training.eval_model

Output

Training will produce:

xlmr_multitask_sinhala.pt


Copy this into your backend:

bias-aware-scoring-engine/app/models/

Purpose

This repo is only for training.
The backend repo handles:

Scoring

Fairness (SPD, DIR, EOD)

Bias mitigation

REST API

This training repo provides the final transformer model used by the backend.