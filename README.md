# Predicting Heart Disease Using Machine Learning

This project explores various Python-based machine learning and data science libraries to build a predictive model capable of determining whether a patient has heart disease based on their clinical attributes.

---

## 1. Problem Definition
Given clinical parameters about a patient, can we accurately predict whether or not they have heart disease?

---

## 2. Data
* The original data originates from the **Cleveland Heart Disease dataset** available in the UCI Machine Learning Repository.
* A version of this dataset is also hosted on Kaggle.

---

## 3. Features & Data Dictionary
The dataset consists of 303 rows and 14 columns (features). Below is the data dictionary for each feature:

1. **`age`**: Age in years
2. **`sex`**: Sex (`1` = male; `0` = female)
3. **`cp`**: Chest pain type
   * `0`: Typical angina (chest pain related to decreased blood supply to the heart)
   * `1`: Atypical angina (chest pain not related to the heart)
   * `2`: Non-anginal pain (typically esophageal spasms, non-heart related)
   * `3`: Asymptomatic (chest pain not showing signs of disease)
4. **`trestbps`**: Resting blood pressure (in mm Hg on admission to the hospital) — values above 130–140 are typically cause for concern.
5. **`chol`**: Serum cholesterol in mg/dl (Serum = $LDL + HDL + 0.2 \times triglycerides$) — values above 200 are cause for concern.
6. **`fbs`**: Fasting blood sugar > 120 mg/dl (`1` = true; `0` = false) — values > 126 mg/dL signal diabetes.
7. **`restecg`**: Resting electrocardiographic results
   * `0`: Nothing to note
   * `1`: ST-T wave abnormality (signals non-normal heartbeat)
   * `2`: Possible or definite left ventricular hypertrophy (enlarged heart main pumping chamber)
8. **`thalach`**: Maximum heart rate achieved
9. **`exang`**: Exercise-induced angina (`1` = yes; `0` = no)
10. **`oldpeak`**: ST depression induced by exercise relative to rest
11. **`slope`**: The slope of the peak exercise ST segment
    * `0`: Upsloping
    * `1`: Flat sloping (typical healthy heart)
    * `2`: Downsloping (signs of an unhealthy heart)
12. **`ca`**: Number of major vessels (0–3) colored by fluoroscopy
13. **`thal`**: Thallium stress result (`1`, `3` = normal; `6` = fixed defect; `7` = reversible defect)
14. **`target`**: Have disease or not (`1` = yes, `0` = no) — **the predicted attribute**

---

## 4. Preparing the Tools & Libraries
The project relies on standard Python libraries for Exploratory Data Analysis (EDA), plotting, and machine learning modeling via Scikit-Learn:

* **Data Handling & Math:** `numpy`, `pandas`
* **Visualization:** `matplotlib.pyplot`, `seaborn`
* **Machine Learning Models:** 
  * `LogisticRegression`
  * `KNeighborsClassifier`
  * `RandomForestClassifier`
* **Model Evaluation & Validation:** `train_test_split`, `cross_val_score`, `RandomizedSearchCV`, `GridSearchCV`, evaluation metrics (`confusion_matrix`, `classification_report`, `precision_score`, `recall_score`, `f1_score`, `RocCurveDisplay`).

---

## 5. Exploratory Data Analysis (EDA) Highlights
* **Dataset Shape:** 303 samples, 14 attributes.
* **Missing Values:** Checked using `df.isna().sum()` (confirmed clean with 0 missing values).
* **Target Distribution:** The dataset contains a balanced distribution of target classes (~165 positive vs. 138 negative cases).
* **Visualizations included in the workflow:**
  * Target frequency bar chart.
  * Heart disease frequency breakdown by `sex`.
  * Scatter plots analyzing `age` vs. `thalach` (Max Heart Rate) grouped by disease status.
  * Chest pain type (`cp`) frequency comparisons.
  * Full feature correlation matrix heatmap.

---

## 6. Getting Started / Running the Code
To load and run the dataset locally in your environment, update the file path in the data loading block to match your local file directory:

```python
import pandas as pd

df = pd.read_csv("path/to/your/heart-disease.csv")
df.shape
