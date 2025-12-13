# Phase 5 – Transformer-based Sinhala Essay Scoring (XLM-RoBERTa)

## Model
- **Architecture:** XLM-RoBERTa Large
- **Fine-tuning strategy:** Partial fine-tuning (last 4 encoder layers)
- **Heads:** Multi-head regression (Richness, Organization, Technical, Total)

## Dataset
- **Total essays:** 136
- **Data type:** Synthetic (ChatGPT-generated essays)
- **Scoring:** Synthetic (ChatGPT-assigned rubric scores)
- **Language:** Sinhala
- **Validation split:** ~20 essays (held-out)

## Training Summary
- Training loss converged smoothly from **1.17 → 0.42**
- No loss divergence or instability observed
- Small oscillations due to multi-head regression and limited dataset size
- Indicates stable optimization and absence of severe overfitting

## Evaluation Results (Validation Set)
- **MAE:** 1.00
- **RMSE:** 1.23
- **Pearson Correlation (r):** 0.58

## Interpretation
- The model demonstrates moderate agreement with synthetic grader scores
- Average prediction error is approximately ±1 mark on a 14-point scale
- Correlation indicates the model captures relative ranking trends among essays
- Results are reasonable given the limited and synthetic nature of the dataset

## Limitations
- Dataset consists entirely of synthetic essays and scores
- Results are intended to validate architectural feasibility rather than real-world grading accuracy
- Performance is expected to improve with human-annotated teacher scores

## Conclusion
This phase validates the feasibility of a transformer-based multi-head regression model for Sinhala essay scoring. The system demonstrates stable training behavior, reasonable generalization on unseen data, and readiness for integration with downstream components and future re-training on real teacher-scored essays.
