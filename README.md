[README_PROJECT.md](https://github.com/user-attachments/files/29831513/README_PROJECT.md)
Project Report — Network Intrusion Detection using GNN (UNSW-NB15)

1. Project: Build a Graph Neural Network (GNN) based Network Intrusion Detection (NID) system using the UNSW-NB15 dataset.
2. Dataset: UNSW-NB15 (training, validation, testing). Raw CSVs are in `dataset/raw/`.
3. Goal: Detect and classify network attacks (binary and multiclass), optimize detection rate and F1-score.
4. Preprocessing: feature cleaning, normalization, label mapping, and graph construction. Processed datasets saved in `dataset/processed/`.
5. Model: GNN architecture implemented in `utils/model.py`. Use PyTorch / PyTorch Geometric for graph operations and training.
6. Training: Scripts available: `nb15_main.py`. Training produces model artifacts in `model/01 - UNSW-NB15/`.
7. Evaluation: Use accuracy, precision, recall, F1-score, confusion matrix, and per-class detection rates. Visualizations saved under `image/01 - UNSW-NB15/`.
8. Results & Logs: Training and evaluation logs are stored in `log/01 - UNSW-NB15/`. Model checkpoints and final outputs are in `model/`.
9. Reproducibility: Install dependencies from `requirements.txt`. Ensure dataset CSVs are present in `dataset/raw/` before running preprocessing.
10. How to run:

    python nb15_pre_processing.py
    python nb15_main.py --config config.yml

11. Future work: hyperparameter search, class imbalance handling, explainability (feature importance), and deployment as a real-time detector.

Contact: Maintainer — check repository README for author details.
