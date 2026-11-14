# Classification and Fraud Detection for Sensitive Documents

Machine learning model that detects tampered identity documents (passports and driver's licenses) using forensic image features. Built for **Verizon** through the **Break Through Tech AI Studio** (Fall 2026).

---

## 👥 Team Members

| Name | GitHub Handle | Contribution |
|---|---|---|
| Yuting Wu | [@yuting-0](https://github.com/yuting-0) | [e.g., data preprocessing, feature extraction, model training] |
| Drashti Shah | [@Dshah1003](https://github.com/Dshah1003) | [e.g., feature engineering, model comparison, evaluation, documentation] |
| Kasish Jain | [@handle] | [contribution] |

---

## 🎯 Project Highlights

- Built a binary classifier that separates genuine from tampered identity documents in the IDNet dataset
- Engineered **295 forensic image features** (including HOG and LBP) to capture signs of tampering
- Compared **13 models**; **AdaBoost** performed best, with **91.5% 5-fold cross-validation accuracy** and **95% recall on tampered documents**
- High recall on fakes means very few fraudulent documents get through, which is the priority for an identity-verification use case

---

## 🏗️ Project Overview

**Program:** This project is part of the Break Through Tech AI Program's AI Studio, where student teams solve a real business problem for a host company.

**Host company:** Verizon

**Objective:** Identity verification systems have to detect forged or edited documents. Our goal is to build a model that classifies a document image as genuine or tampered.

**Scope (milestones):**
1. Binary classification: genuine vs. tampered (September)
2. An autonomous AI agent built around the classifier (October)
3. Multi-class classification of the tampering type (November)

**Impact:** Automated fraud detection reduces manual review and helps stop identity fraud during account sign-up and verification.

---

## 📊 Data Exploration

**Dataset:** [IDNet](link-to-dataset), a synthetic identity document dataset for fraud detection.

- The Kaggle mirror contains only part of the full dataset: West Virginia driver's licenses, Greek passports and Russian passports
- **Working subset:** 600 images, 300 genuine and 300 tampered
- **Tampering methods (5):** full blur, partial blur, copy-move, splice, text overwrite

**Preprocessing:** [e.g., resizing, grayscale conversion, normalization]

**Challenges:**
- The full dataset is very large, which slowed downloading and delayed the first milestone
- The Kaggle mirror covers fewer document types than the full dataset

**Visualizations:**
<!-- Add images saved in /images -->
![Sample genuine vs tampered documents](images/sample_documents.png)
![Class distribution](images/class_distribution.png)

---

## 🧠 Model Development

**Features:** 295 handcrafted forensic features, including:
- HOG (Histogram of Oriented Gradients): edge and shape structure
- LBP (Local Binary Patterns): local texture changes
- [other feature groups, e.g., noise, color, compression artifacts]

**Models compared (13):** [list, e.g., Logistic Regression, SVM, Random Forest, Gradient Boosting, AdaBoost, KNN, ...]

**Training setup:**
- 5-fold cross-validation
- Metrics: accuracy, precision, recall and F1, with a focus on **recall for the tampered class**
- [hyperparameter tuning method, e.g., GridSearchCV]

---

## 📈 Results & Key Findings

| Model | CV Accuracy | Recall (Tampered) | F1 |
|---|---|---|---|
| **AdaBoost** | **91.5%** | **95%** | [ ] |
| [Model 2] | [ ] | [ ] | [ ] |
| [Model 3] | [ ] | [ ] | [ ] |

**Key findings:**
- AdaBoost gave the best balance of accuracy and fraud recall
- [Which features mattered most, e.g., LBP texture features]
- [Which tampering type was hardest to detect]

![Confusion matrix](images/confusion_matrix.png)
![Feature importance](images/feature_importance.png)

---

## 👩🏽‍💻 Setup and Installation

```bash
# 1. Clone the repository
git clone https://github.com/Break-Through-Tech/Telecom-2B-classification-and-fraud-detection-for-sensitive-documents.git
cd Telecom-2B-classification-and-fraud-detection-for-sensitive-documents

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dataset:**
- A small sample for testing is in `data/sample/`
- Full dataset: [Kaggle / Zenodo link]. Download it and place it in `data/raw/`

**Run the project:**
```bash
# Extract features
python [src/extract_features.py]

# Train models
python [src/train.py]

# Evaluate
python [src/evaluate.py]
```

Or open the notebooks in order:
1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_model_training_evaluation.ipynb`

---

## 📁 Repository Structure

```
├── data/
│   └── sample/          # small sample dataset for testing
├── notebooks/           # annotated Jupyter notebooks
├── src/                 # feature extraction, training, evaluation scripts
├── images/              # plots used in this README
├── requirements.txt
└── README.md
```

---

## 🚀 Next Steps

- **Limitations:** trained on a small subset (600 images) covering only 3 document types, so it may not generalize to other countries or states
- Train on the full IDNet dataset from Zenodo
- Try CNNs or transfer learning (e.g., ResNet, EfficientNet) and compare them with handcrafted features
- Finish multi-class classification of the tampering type
- Build the AI agent workflow around the classifier

---

## 📝 License

This project is licensed under the [MIT License](LICENSE). [Confirm with the Challenge Advisor]

---

## 📄 References

- IDNet dataset: [paper/link]
- [Other papers or tutorials used]

---

## 🙏 Acknowledgements

Thanks to our Challenge Advisor, our AI Studio Coach, the Verizon team and Break Through Tech for their support.
