# TruthLens-A-Hybrid-Machine-Learning-and-Transformer-Framework-for-Fake-News-Detection
An end-to-end NLP pipeline that benchmarks Classical Machine Learning, Embedding-based models, and Fine-Tuned Transformers to detect fake news ,balancing accuracy, interpretability, and real-world scalability.

# Project
As a data scientist passionate about socially responsible AI, I built TruthLens to explore one key question:

“How do traditional ML methods compare to modern Transformer models in detecting fake news?”

This project evolves across three generations of NLP modeling, demonstrating how each step - from interpretable baselines to deep contextual models - improves accuracy and reliability.

# What the project demonstrates
What the Project Demonstrates

- A full AI experimentation pipeline - Classical ML -> Embeddings -> Transformers.
- How model complexity vs. interpretability affects real-world usability.
- Transparent evaluation with explainable components (TF-IDF & SHAP).
- A realistic benchmarking study mirroring production-grade workflows.

# Project Overview
| Stage       | Model Family           | Key Models                                    | Goal                                |
| ----------- | ---------------------- | --------------------------------------------- | ----------------------------------- |
| **Stage 1** | Classical ML           | Logistic Regression · XGBoost · Random Forest | Build fast, interpretable baselines |
| **Stage 2** | Neural Embeddings      | Sentence-BERT · E5 + MLP                      | Capture semantic similarity         |
| **Stage 3** | Fine-Tuned Transformer | RoBERTa-base                                  | Learn deep contextual nuances       |


# Workflow 

1. Data Collection (PolitiFact + BuzzFeed)
2. Text Cleaning & Preprocessing
  - Classical ML (TF-IDF + LogReg/XGBoost)
  - Embeddings (Sentence-BERT / E5 + MLP)
  - Transformer Fine-Tuning (RoBERTa-base)


# Tech Stack

Languages: Python

Libraries: scikit-learn · XGBoost · PyTorch · Hugging Face Transformers · Sentence-Transformers · Pandas · NumPy · Matplotlib · SHAP
Environment: Google Colab (NVIDIA T4 GPU)

# Dataset
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# Results
| Model Family     | Representative Model          | **Macro F1 (Test)** | Key Strength                     |
| ---------------- | ----------------------------- | ------------------: | -------------------------------- |
| **Classical ML** | Logistic Regression (TF-IDF)  |            **0.45** | Fast & interpretable baseline    |
| **Embeddings**   | SBERT + MLP                   |            **0.50** | Handles paraphrasing & semantics |
| **Transformer**  | **RoBERTa-base (fine-tuned)** |            **0.55** | Deep contextual understanding    |

Progressive improvement from 0.45 - 0.55 F1, demonstrating the value of fine-tuning over traditional models.

# Insights

Logistic Regression serves as a strong, lightweight baseline for scalable deployment.

SBERT + MLP bridges lexical gaps, improving recall on reworded claims.

RoBERTa-base excels at detecting contextual inconsistencies in political or emotional tone.

Explainability through TF-IDF coefficients + SHAP ensures transparent decision-making.

Mimics real-world AI lifecycle development - prototype -> optimize -> interpret -> deploy.

# Skills Demonstrated
| Area                    | Tools / Techniques                                   |
| ----------------------- | ---------------------------------------------------- |
| **Machine Learning**    | Logistic Regression · SVM · Random Forest · XGBoost  |
| **Deep Learning & NLP** | Sentence-BERT · E5 Embeddings · RoBERTa Fine-Tuning  |
| **Explainable AI**      | SHAP · Feature Coefficients                          |
| **Evaluation Metrics**  | Macro-F1 · ROC-AUC · PR-AUC                          |
| **Engineering**         | Model Pipelines · Experiment Tracking · GPU Training |


# Future Work

Integrate Retrieval-Augmented Generation (RAG) for evidence-based verification.

Build a Streamlit dashboard for live fake-news prediction.

Extend to multilingual and multimodal datasets.

Experiment with DeBERTa v3 or DistilRoBERTa for lighter models.










