# AI & ML Internship – Task 1

## 📌 Objective
Learn how to clean and preprocess raw data for ML using the Titanic dataset.

## ✅ Steps Performed
1. Imported dataset and explored basic info (nulls, data types).
2. Handled missing values:
   - Age → filled with median
   - Embarked → filled with mode
   - Dropped Cabin (too many nulls)
3. Converted categorical features into numerical:
   - Sex → Label Encoding
   - Embarked → One-Hot Encoding
4. Standardized numerical features (Age, Fare) using StandardScaler.
5. Visualized outliers with boxplots and removed extreme values using IQR method.

## 🛠 Tools & Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📂 Repo Structure
```
AI-ML-Internship-Task1/
│── data/                # Dataset
│── screenshots/         # Visual outputs
│── task1_preprocessing.ipynb
│── README.md
```

