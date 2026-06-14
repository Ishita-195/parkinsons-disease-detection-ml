# Parkinson's Disease Detection — Machine Learning System

A machine learning project for early detection of Parkinson's Disease using biomedical voice measurements. Includes exploratory data analysis, preprocessing, model training, hyperparameter tuning, and an interactive **Streamlit** web application.

---

##  Project Structure

```
parkinsons-disease-detection-ml/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks (EDA, preprocessing, modeling)
├── models/             # Saved trained models
├── app/                # Streamlit web application
├── results/            # Evaluation metrics, plots, outputs
└── .gitignore
```

---

##  Overview

Parkinson's Disease (PD) is a progressive neurological disorder. This project uses machine learning on biomedical voice measurements to classify whether a patient has Parkinson's Disease.

**Key stages:**
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Training (multiple classifiers)
- Hyperparameter Tuning
- Interactive Streamlit Web App for real-time prediction

---

##  Dataset

The dataset used is the **UCI Parkinson's Disease dataset**, containing biomedical voice measurements from 195 recordings (147 with PD, 48 healthy controls).

**Key features include:**
- MDVP:Fo(Hz) — Average vocal fundamental frequency
- MDVP:Jitter(%), MDVP:Shimmer — Measures of variation in frequency and amplitude
- NHR, HNR — Noise-to-harmonics ratio
- RPDE, DFA — Nonlinear dynamical complexity measures
- PPE — Pitch period entropy

**Target column:** `status` (1 = Parkinson's, 0 = Healthy)

---

##  Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost / Gradient Boosting

Hyperparameter tuning was performed using **GridSearchCV / RandomizedSearchCV**.

---

##  Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Ishita-195/parkinsons-disease-detection-ml.git
cd parkinsons-disease-detection-ml

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
cd app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Exploratory data analysis and visualizations |
| `02_Preprocessing.ipynb` | Data cleaning, scaling, feature selection |
| `03_ModelTraining.ipynb` | Training and evaluating ML models |
| `04_HyperparameterTuning.ipynb` | Optimizing model performance |

---

## 📈 Results

Model performance is tracked using:
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix

Evaluation outputs are saved in the `results/` folder.

---

## 🖥️ Streamlit App

The web app allows users to input voice measurement values and get an instant prediction on whether the patient is likely to have Parkinson's Disease. The best-performing model is loaded from the `models/` directory.

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas, NumPy** — Data manipulation
- **Matplotlib, Seaborn** — Visualization
- **Scikit-learn** — ML models and preprocessing
- **XGBoost** — Gradient boosting
- **Streamlit** — Web application
- **Joblib / Pickle** — Model serialization
- **Jupyter Notebook** — Experimentation

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Ishita Anand**  
Computer Science & Engineering Student | KIIT University, Bhubaneswar  
GitHub: [@Ishita-195](https://github.com/Ishita-195)
---

> ⚠️ *This project is for educational purposes only and is not intended for clinical or diagnostic use.*
