# 🚢 Titanic Survival Prediction — AI Project

[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/titanic)
[![GitHub](https://img.shields.io/badge/GitHub-Titanic__Machine__Learning-181717?logo=github&logoColor=white)](https://github.com/msabr/Titanic_Machine_Learning)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Predict which passengers survived the Titanic shipwreck using machine learning — a classic binary classification challenge from Kaggle.

**Academic project (2025–2026)** supervised by **Prof. N. ABOUTABIT**  

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Features & Preprocessing](#-features--preprocessing)
- [Models & Validation](#-models--validation)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Reports](#-reports)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 Overview

This project tackles the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic), one of the most popular introductory machine learning challenges. The goal is to build a predictive model that answers:

> *"What sorts of people were more likely to survive the Titanic disaster?"*

The repository implements a complete ML pipeline:
- Exploratory Data Analysis (EDA)
- Missing value handling & preprocessing
- Feature engineering
- Model training + evaluation
- Prediction generation and submission export

---

## 📊 Dataset

The dataset is provided by Kaggle and contains information about **891 passengers (train)** and **418 passengers (test)**.

| File | Description |
|------|-------------|
| `data/train.csv` | Training data with survival labels |
| `data/test.csv` | Test data for prediction submission |
| `data/gender_submission.csv` | Sample submission file |

### Key Features

| Feature | Description |
|---------|-------------|
| `Survived` | Target variable (0 = No, 1 = Yes) |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C, Q, S) |

---

## 📁 Project Structure

> This reflects the **current repository layout**.

```text
Titanic_Machine_Learning/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── src/
│   ├── data/              # loading / preprocessing modules
│   ├── features/          # feature engineering
│   ├── models/            # training / evaluation / optimization
│   ├── utils/             # config + helpers
│   └── visualization/     # EDA plots
├── main.py                # runs the full pipeline end-to-end
├── REPORT.md              # detailed report (markdown)
├── DELIVERABLES.md        # checklist of deliverables
├── Rapport_Project_AI.pdf # full technical report (PDF)
└── requirements.txt
```

---

## ⚙️ Features & Preprocessing

Key engineering steps used in the pipeline include:

- **Missing value imputation**
  - `Age`: imputation strategy based on passenger groups (e.g., by class/sex)
  - `Embarked`: mode imputation
  - `Fare`: median imputation (mainly affects test set)

- **Feature engineering**
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone` flag
  - `Title` extracted from `Name` (Mr, Mrs, Miss, Master, Rare, ...)
  - `HasCabin` (Cabin present vs missing)

- **Encoding**
  - One-hot encoding for categorical features (Sex, Embarked, Title, etc.)

- **Scaling**
  - `StandardScaler` applied before model training (see `main.py`)

---

## 🤖 Models & Validation

The project trains and evaluates classification models (baseline + optimized) and uses:

- **Train/Validation split** (stratified)
- **Cross-validation**
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC

> For the exact implementation details and metrics, see `REPORT.md`.

---

## 📈 Results

A short summary (update with your Kaggle score/rank):

| Item | Value |
|------|------:|
| Best local validation accuracy | ~80–82% (varies by model/settings) |
| ROC-AUC | ~0.84–0.86 |
| Kaggle public score | _Add your score_ |
| Kaggle rank | _Add your rank / Top X%_ |

---

## 🛠️ Installation

**Prerequisites:** Python 3.9+

```bash
# 1) Clone the repository
git clone https://github.com/msabr/Titanic_Machine_Learning.git
cd Titanic_Machine_Learning

# 2) (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

> Note: `requirements.txt` currently lists core libraries (pandas/numpy/sklearn/matplotlib/seaborn).

---

## 🚀 Usage

### Run the full pipeline (recommended)

```bash
python main.py
```

This will run the pipeline end-to-end (EDA → preprocessing → training → evaluation → predictions).

### Data location

Make sure the Kaggle CSV files exist in:

```text
data/train.csv
data/test.csv
```

---

## 📑 Reports

- `REPORT.md` — detailed methodology, preprocessing, evaluation, and recommendations  
- `Rapport_Project_AI.pdf` — full technical report (PDF)  
- `DELIVERABLES.md` — deliverables summary / checklist

---

## 🤝 Contributing

Contributions are welcome:

1. Fork the repository
2. Create a branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add improvement"`)
4. Push (`git push origin feature/improvement`)
5. Open a Pull Request

---

- **Team**: Mohamed SABR, Abdejlil SALMI, Soufaine ZEKAOUI, Anass LAMHADAR, Aymane ELMOUD
