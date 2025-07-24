# 🧬 Antimicrobial Peptide Deduction

This project focuses on the classification of antimicrobial peptides (AMPs) using machine learning techniques and sequence data from the **DADP** (Database of Antimicrobial Peptides).

## 📌 Project Summary

- Directed a research project on **2,150 peptide sequences** from the DADP dataset, uncovering **1,923 unique bioactive sequences**.
- Engineered a classification model using **supervised learning**, leveraging advanced **data preprocessing** and **feature selection** to enhance model performance.
- Delivered a **25% increase in predictive accuracy** and an overall **80% accuracy** using precision, recall, and F1-score metrics.
- Provides a tool for researchers and developers to accelerate **antimicrobial peptide discovery and optimization**.

---

## 📁 Project Structure

```
Antimicrobial-Peptide-Deduction/
│
├── data/                     # Raw and processed datasets
│   └── DADP/                 # Original peptide sequences
│
├── notebooks/                # Jupyter notebooks for EDA and modeling
│
├── src/                      # Source code and pipeline scripts
│   ├── preprocess.py         # Data cleaning & feature extraction
│   ├── train_model.py        # Model training
│   ├── evaluate.py           # Evaluation metrics
│   └── utils.py              # Helper functions
│
├── outputs/                  # Trained models, metrics, results
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 How to Run This Project

### 1. 🔧 Installation

```bash
git clone https://github.com/Abdulsamad01/Antimicrobial-Peptide-Deduction.git
cd Antimicrobial-Peptide-Deduction
pip install -r requirements.txt
```

> ✅ Ensure Python 3.8+ is installed.

---

### 2. 📦 Dataset Preparation

Place your peptide dataset (e.g., `peptides.csv`) inside the `data/DADP/` directory.

If your dataset is not included, you can download it from the official [DADP website](https://www.camp.bicnirrh.res.in/).

---

### 3. 🧹 Preprocess the Data

```bash
python src/preprocess.py \
  --input data/DADP/peptides.csv \
  --output data/processed/features.pkl
```

This script will:
- Clean peptide sequences
- Extract and encode features
- Save the feature matrix for modeling

---

### 4. 🧠 Train the Model

```bash
python src/train_model.py \
  --features data/processed/features.pkl \
  --model-output outputs/model.pkl
```

- Supports classifiers like Random Forest, SVM, etc.
- Hyperparameters can be tuned inside the script

---

### 5. 📈 Evaluate Model Performance

```bash
python src/evaluate.py \
  --model outputs/model.pkl \
  --features data/processed/features.pkl \
  --output outputs/metrics.json
```

- Outputs precision, recall, F1-score, accuracy
- Saves the results in a JSON file

---

## 💡 Example Usage (in Python)

```python
from src.predict import predict_sequence

model_path = "outputs/model.pkl"
sequence = "GIGKFLKKAKKFGKAFVKILKK"

prediction = predict_sequence(model_path, sequence)
print(f"Predicted AMP probability: {prediction:.2f}")
```

---

## 📊 Performance Overview

| Metric      | Value       |
|-------------|-------------|
| Accuracy    | ~80%        |
| Precision   | High        |
| Recall      | High        |
| F1-Score    | High        |

- ⚙️ Achieved using robust feature engineering and classifier tuning
- 🧬 25% boost in prediction accuracy compared to baseline

---

## 🧪 Technologies Used

- Python (NumPy, pandas, scikit-learn)
- Jupyter Notebooks
- Feature selection & encoding methods
- Supervised ML algorithms (e.g., Random Forest, SVM)

---

## 🧰 Reproducibility

To ensure reproducible results, random seeds and model configurations are fixed inside each script.

---

## 🙌 Contributing

Feel free to fork this repository, submit pull requests, or create issues for:

- Adding new ML algorithms
- Enhancing feature engineering
- Improving evaluation workflows

---

## 📬 Contact

**Author:** Abdulsamad  
**Email:** [your-email@example.com]  
**GitHub:** [https://github.com/Abdulsamad01](https://github.com/Abdulsamad01)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
