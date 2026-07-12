# MMShield: Multimodal Prompt Injection Detection Framework

MMShield is a research-oriented framework for detecting multimodal prompt injection attacks against Vision-Language Models (VLMs). The project focuses on identifying malicious instructions embedded in documents using multiple detection techniques.

## Features

- Typographic Attack Detection
- Adversarial Patch Detection
- Steganographic Attack Detection (In Progress)
- Candidate Region Generation
- CNN Feature Extraction (EfficientNet-B0)
- Handcrafted Visual Features
- Feature Fusion
- XGBoost-based Threat Classification
- Visualization of Predictions and Threat Heatmaps

---

## Project Structure

```
MMShield/
│
├── adversarial/
│   ├── attack_generator.py
│   ├── candidate_generator.py
│   ├── feature_extractor.py
│   ├── receipt_features.py
│   ├── dataset_builder.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visualize.py
│   └── config.py
│
├── typographic/
├── datasets/
├── patches/
├── docs/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Adversarial Patch Detection Pipeline

```
Receipt Image
      │
      ▼
Attack Generator
      │
      ▼
Candidate Generator
      │
      ▼
CNN Feature Extraction
      │
      ▼
Handcrafted Feature Extraction
      │
      ▼
Feature Fusion
      │
      ▼
XGBoost Classifier
      │
      ▼
Prediction
      │
      ▼
Visualization
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd mmshield
```

Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

### Apple Silicon (M1/M2/M3/M4)

If XGBoost raises an OpenMP-related error:

```bash
brew install libomp

export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
```

---

## Running the Pipeline

### 1. Generate Adversarial Samples

```bash
python adversarial/attack_generator.py
```

### 2. Generate Candidate Regions

```bash
python adversarial/candidate_generator.py
```

### 3. Extract CNN Features

```bash
python adversarial/feature_extractor.py
```

### 4. Generate Handcrafted + Fused Features

```bash
python adversarial/receipt_features.py
```

### 5. Build Dataset

```bash
python adversarial/dataset_builder.py
```

### 6. Train Models

```bash
python adversarial/train.py
```

### 7. Evaluate

```bash
python adversarial/evaluate.py
```

### 8. Predict

```bash
python adversarial/predict.py
```

### 9. Visualize

```bash
python adversarial/visualize.py
```

---

## Experimental Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **99.05%** |
| Precision | **87.10%** |
| Recall    | **90.00%** |
| F1 Score  | **88.52%** |
| ROC-AUC   | **99.82%** |

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- Scikit-learn
- XGBoost
- NumPy
- Pandas
- Matplotlib

---

## Current Status

- ✅ Typographic Detection
- ✅ Adversarial Patch Detection
- 🚧 Steganographic Detection (In Progress)

---

## Future Work

- Complete steganographic attack detector
- Integrate all modules into a unified multimodal detection framework
- Extend support to additional Vision-Language Models
- Improve real-world robustness against unseen attacks

---

## License

This project is developed for academic and research purposes.
