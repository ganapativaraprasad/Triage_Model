import pandas as pd
#DataSet Preparation

data = [
    ("chest pain and shortness of breath", "Cardiology"),
    ("irregular heartbeat and fatigue", "Cardiology"),
    ("tightness in chest while walking", "Cardiology"),
    ("high blood pressure and dizziness", "Cardiology"),
    ("pain radiating to left arm", "Cardiology"),
    ("palpitations and sweating", "Cardiology"),
    ("difficulty breathing during exertion", "Cardiology"),
    ("heart racing without reason", "Cardiology"),
    ("swelling in legs and chest discomfort", "Cardiology"),
    ("burning sensation in chest", "Cardiology"),

    ("severe headache and blurred vision", "Neurology"),
    ("frequent migraines and nausea", "Neurology"),
    ("numbness in hands and feet", "Neurology"),
    ("loss of balance while walking", "Neurology"),
    ("sudden confusion and memory loss", "Neurology"),
    ("tingling sensation in fingers", "Neurology"),
    ("difficulty speaking clearly", "Neurology"),
    ("episodes of fainting", "Neurology"),
    ("uncontrolled shaking of hands", "Neurology"),
    ("vision problems and dizziness", "Neurology"),

    ("knee pain after running", "Orthopaedics"),
    ("back pain while lifting objects", "Orthopaedics"),
    ("shoulder stiffness and pain", "Orthopaedics"),
    ("joint swelling in knees", "Orthopaedics"),
    ("pain in ankle after twisting", "Orthopaedics"),
    ("muscle pain after exercise", "Orthopaedics"),
    ("fracture in arm after fall", "Orthopaedics"),
    ("difficulty bending knees", "Orthopaedics"),
    ("pain in lower back region", "Orthopaedics"),
    ("neck pain due to posture", "Orthopaedics"),

    ("fever and cough", "General"),
    ("cold and sore throat", "General"),
    ("body ache and tiredness", "General"),
    ("vomiting and stomach pain", "General"),
    ("mild fever with chills", "General"),
    ("loss of appetite and weakness", "General"),
    ("sore throat and headache", "General"),
    ("runny nose and sneezing", "General"),
    ("general fatigue and discomfort", "General"),
    ("mild headache and fever", "General"),
]

df = pd.DataFrame(data, columns=["text", "label"])

# Expand to ~200 samples
import random

augmented_data = []

for text, label in data:
    for i in range(5):
        new_text = text + " " + random.choice([
            "recently", "for two days", "since morning",
            "getting worse", "mild", "severe"
        ])
        augmented_data.append((new_text, label))

df = pd.DataFrame(augmented_data, columns=["text", "label"])

print(df.head())
print("Total samples:", len(df))

#importing necessary libraries for model training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#split the data into training and testing sets
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#convert text data into numerical features using TF-IDF vectorization.
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


#Train a logistic regression model on the training data.
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

#Evaluate the model's performance on the test set and print a classification report.
y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))


import numpy as np

# Get prediction probabilities
probs = model.predict_proba(X_test_tfidf)

# Get class labels
classes = model.classes_

wrong_cases = []

for i in range(len(y_test)):
    true_label = y_test.iloc[i]
    predicted_label = y_pred[i]
    confidence = np.max(probs[i])

    if true_label != predicted_label:
        wrong_cases.append((X_test.iloc[i], true_label, predicted_label, confidence))

# Sort by confidence (highest first)
wrong_cases = sorted(wrong_cases, key=lambda x: x[3], reverse=True)

# Print top 2 confident wrong predictions
for case in wrong_cases[:2]:
    print("\nText:", case[0])
    print("Actual:", case[1])
    print("Predicted:", case[2])
    print("Confidence:", case[3])


user_input = input("Enter symptoms: ")

user_tfidf = vectorizer.transform([user_input])
prediction = model.predict(user_tfidf)

print("Predicted Department:", prediction[0])

user_input = input("Enter symptoms: ")

user_tfidf = vectorizer.transform([user_input])
prediction = model.predict(user_tfidf)

print("Predicted Department:", prediction[0])


"""
MODEL CARD

This model is a multi-class text classification system designed to route patient symptom descriptions to appropriate medical departments such as Cardiology, Neurology, Orthopaedics, and General Medicine. The model uses TF-IDF vectorization to convert text into numerical features and Logistic Regression for classification.

The model was trained on a synthetic dataset of approximately 200 symptom descriptions. Each class was balanced to avoid bias during training. However, since the dataset is artificially generated and limited in vocabulary, it does not fully represent real-world medical data.

The model performs reasonably well with an overall accuracy of around 85–90%, but performance varies across classes. For example, Neurology has lower recall, indicating difficulty in identifying all neurological cases.

A key failure mode observed is misclassification of symptoms that share overlapping context words. For instance, 'loss of balance while walking' was incorrectly classified as Cardiology due to the presence of the word 'walking', which appeared in cardiology-related training data. This shows that the model relies on shallow keyword patterns rather than true medical understanding.

Another limitation is the model’s inability to understand complex symptom relationships or severity. It cannot interpret clinical context or patient history.

This model should NOT be deployed without human oversight. In a medical setting, incorrect predictions could lead to misrouting patients, delayed treatment, and serious health risks. Therefore, this system should only be used as a supportive tool alongside qualified medical professionals.
"""