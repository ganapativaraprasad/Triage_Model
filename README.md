# 🏥 Medical Triage System using NLP

## 📌 Overview
This project is a simple NLP-based machine learning system that classifies patient symptom descriptions into appropriate medical departments:
- General
- Cardiology
- Neurology
- Orthopaedics

---

## 🎯 Objective
To build a multi-class text classification model using TF-IDF and Logistic Regression that can assist in routing patients based on symptoms.

---

## 🧠 Approach

1. **Dataset Creation**
   - Created a synthetic dataset of 200 samples
   - Balanced across all classes

2. **Text Processing**
   - Converted text into numerical features using TF-IDF

3. **Model Training**
   - Used Logistic Regression for classification

4. **Evaluation**
   - Evaluated using per-class F1-score

5. **Error Analysis**
   - Identified misclassifications and analyzed model weaknesses

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 📊 Results
- Achieved ~88% accuracy
- Good performance across most classes
- Some confusion in Neurology predictions

---

## ⚠️ Limitations
- Uses synthetic dataset (not real medical data)
- Limited vocabulary
- Cannot understand complex medical context

---

## 🛑 Important Note
This model should NOT be used in real-world medical decision-making without human supervision.

---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install pandas scikit-learn
```
2. Run the script:
```bash
python triage_model.py
```

2. Run the script:
```bash
python triage_model.py
```
