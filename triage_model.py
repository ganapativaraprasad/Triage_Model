import pandas as pd
import numpy as np

# -------------------------------
# STEP 1: LOAD DATASET
# -------------------------------
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

print("Dataset shape:", df.shape)
print(df.head())


# -------------------------------
# STEP 2: FAST CONVERT 0/1 → TEXT
# -------------------------------
print("\nCreating text column...")

symptom_columns = df.columns[1:]

df["text"] = (
    df[symptom_columns]
    .astype(str)
    .apply(lambda x: " ".join(x.index[x == "1"]), axis=1)
)

print("Text column created successfully!")


# -------------------------------
# STEP 3: MAP DISEASE → DEPARTMENT
# -------------------------------
def map_department(disease):
    disease = str(disease).lower()

    cardio_keywords = ["heart", "cardiac", "hypertension", "angina"]
    neuro_keywords = ["migraine", "epilepsy", "brain", "stroke", "paralysis", "headache"]
    ortho_keywords = ["fracture", "arthritis", "joint", "bone", "knee", "back"]

    if any(word in disease for word in cardio_keywords):
        return "Cardiology"
    elif any(word in disease for word in neuro_keywords):
        return "Neurology"
    elif any(word in disease for word in ortho_keywords):
        return "Orthopaedics"
    else:
        return "General"

df["label"] = df["diseases"].apply(map_department)

print("\nLabel Distribution:")
print(df["label"].value_counts())


# -------------------------------
# STEP 4: PREPARE DATA
# -------------------------------
X = df["text"]
y = df["label"]


# -------------------------------
# STEP 5: TRAIN TEST SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# STEP 6: TF-IDF
# -------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -------------------------------
# STEP 7: MODEL TRAINING
# -------------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# -------------------------------
# STEP 8: EVALUATION
# -------------------------------
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_tfidf)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------------
# STEP 9: ERROR ANALYSIS
# -------------------------------
probs = model.predict_proba(X_test_tfidf)

wrong_cases = []

for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        wrong_cases.append((
            X_test.iloc[i],
            y_test.iloc[i],
            y_pred[i],
            np.max(probs[i])
        ))

wrong_cases = sorted(wrong_cases, key=lambda x: x[3], reverse=True)

print("\nTop Wrong Predictions:")
for case in wrong_cases[:2]:
    print("\nText:", case[0])
    print("Actual:", case[1])
    print("Predicted:", case[2])
    print("Confidence:", case[3])


# -------------------------------
# STEP 10: SAMPLE TEST
# -------------------------------
sample = ["chest pain and difficulty breathing"]

sample_tfidf = vectorizer.transform(sample)
prediction = model.predict(sample_tfidf)

print("\nSample Input:", sample[0])
print("Predicted Department:", prediction[0])


# -------------------------------
# STEP 11: USER INPUT
# -------------------------------
user_input = input("\nEnter symptoms: ")

user_tfidf = vectorizer.transform([user_input])
prediction = model.predict(user_tfidf)

print("Predicted Department:", prediction[0])


# -------------------------------
# MODEL CARD
# -------------------------------
"""
MODEL CARD

This model classifies patient symptoms into medical departments using TF-IDF and Logistic Regression.

Dataset:
Binary symptom dataset where symptoms are converted into text.

Approach:
- Convert symptoms → text
- TF-IDF vectorization
- Logistic Regression classification

Limitations:
- Rule-based mapping
- No deep medical understanding
- Depends on dataset quality

Use:
Not suitable for real-world medical decisions without supervision.
"""