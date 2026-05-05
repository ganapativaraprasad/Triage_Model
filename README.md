# 🏥 Medical Triage System using NLP

## 📌 Overview
This project is an NLP-based machine learning system that classifies patient symptom descriptions into appropriate medical departments:
- General
- Cardiology
- Neurology
- Orthopaedics

---

## 🎯 Objective
To build a multi-class text classification model using TF-IDF and Logistic Regression to assist in routing patients based on symptoms.

---

## 🧠 Approach

1. **Dataset Usage**
   - Used a real-world medical symptom dataset
   - Symptoms were provided in binary (0/1) format across multiple columns

2. **Data Preprocessing**
   - Converted binary symptom columns into readable text format
   - Combined active symptoms into a single text representation

3. **Label Mapping**
   - Mapped diseases to departments (Cardiology, Neurology, Orthopaedics, General)

4. **Feature Engineering**
   - Used TF-IDF vectorization to convert text into numerical features

5. **Model Training**
   - Trained a Logistic Regression classifier

6. **Evaluation**
   - Evaluated model using per-class Precision, Recall, and F1-score

7. **Error Analysis**
   - Identified high-confidence misclassifications
   - Analyzed model limitations

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 📊 Results
- Achieved realistic performance on real dataset
- Performance varies across departments
- Some confusion in overlapping symptom cases (e.g., Neurology vs Cardiology)

---

## 📥 Dataset

The dataset used in this project is **not included** in the repository due to GitHub file size limitations.

👉 Download from Kaggle:
Search: **"Disease Symptoms Dataset"**

After downloading, place it in the project folder:

Final_Augmented_dataset_Diseases_and_Symptoms.csv

---

## ⚠️ Limitations
- Disease-to-department mapping is rule-based
- Model relies on keyword patterns
- Limited understanding of complex medical context

---

## 🛑 Important Note
This model should **NOT** be used for real-world medical decisions without human supervision.

---

## ▶️ How to Run

### 1. Install dependencies
pip install pandas scikit-learn

### 2. Place dataset in project folder

### 3. Run the script
python triage_model.py

---

## 🧪 Sample Prediction

Input:
chest pain and difficulty breathing

Output:
Cardiology

---

## 📄 Author
Prasad