# Titanic ML Project 🚢

Predict Titanic survival using Random Forest on Kaggle dataset.

## Structure

```bash
titanic-ml-project/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv   # Optional baseline from Kaggle
│
├── notebooks/
│   └── titanic_baseline.ipynb  # Main Kaggle notebook or Jupyter notebook
│
├── src/
│   ├── preprocess.py           # Functions for data cleaning & encoding
│   ├── train_model.py          # Model training & evaluation
│   ├── predict.py              # Final predictions for submission
│
├── output/
│   └── submission.csv          # File for Kaggle submission
│
├── README.md                   # Project overview and instructions
├── requirements.txt            # Optional: dependencies if using virtualenv
└── .gitignore                  # Optional: exclude data/output files

```

## Usage

### 1. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Train model:

```bash
python src/train_model.py
```

### 3. Generate predictions:

```bash
python src/predict.py
```
