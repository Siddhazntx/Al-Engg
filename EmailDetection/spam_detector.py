import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD EXTERNAL EMAIL DATASET
# --------------------------------------------------

DATA_PATH = Path("data/spam_ham_dataset.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH.resolve()}\n"
        "Download the CSV from Kaggle and place it "
        "inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

# The Kaggle dataset uses 'text' for email content
# and 'label' for spam/ham classification.
required_columns = {"text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required_columns}. "
        f"Found columns: {list(df.columns)}"
    )

df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.lower().str.strip()

# Retain only the two classes required for this project.
df = df[df["label"].isin(["spam", "ham"])]
df = df[df["text"] != ""]
df = df.drop_duplicates(subset=["text"])

if df["label"].nunique() != 2:
    raise ValueError("Dataset must contain both spam and ham emails.")

print("=" * 55)
print("       EMAIL SPAM DETECTION SYSTEM")
print("=" * 55)

print("\nDATASET INFORMATION")
print("-" * 55)
print(f"Total usable emails: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())

# --------------------------------------------------
# 2. TEXT PREPROCESSING
# --------------------------------------------------

def clean_email(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# --------------------------------------------------
# 4. BUILD THE MACHINE LEARNING PIPELINE
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            preprocessor=clean_email,
            ngram_range=(1, 2),
            max_features=50000
        )
    ),
    ("classifier", MultinomialNB())
])

# --------------------------------------------------
# 5. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining model. Please wait...")
model.fit(X_train, y_train)
print("Model training completed.")

# --------------------------------------------------
# 6. EVALUATE ON UNSEEN TEST DATA
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nMODEL EVALUATION")
print("-" * 55)
print(f"Training emails: {len(X_train)}")
print(f"Testing emails : {len(X_test)}")
print(f"Accuracy       : {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["ham", "spam"],
    target_names=["Legitimate", "Spam"],
    zero_division=0
))

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
))

# --------------------------------------------------
# 7. CLASSIFY A NEW EMAIL
# --------------------------------------------------

def predict_email(email_text):
    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    classes = list(model.named_steps["classifier"].classes_)
    confidence = probabilities[classes.index(prediction)] * 100

    if prediction == "spam":
        result = "SPAM"
    else:
        result = "LEGITIMATE (NOT SPAM)"

    print("\n" + "=" * 55)
    print("EMAIL CLASSIFICATION RESULT")
    print("=" * 55)
    print("Email:", email_text)
    print("Prediction:", result)
    print(f"Model confidence: {confidence:.2f}%")
    print("=" * 55)

    return result


# --------------------------------------------------
# 8. INTERACTIVE EMAIL CHECKER
# --------------------------------------------------

if __name__ == "__main__":
    while True:
        print("\nEnter an email message to classify.")
        print("Type EXIT to close the application.")

        email_text = input("\nEmail: ").strip()

        if email_text.lower() == "exit":
            print("Application closed.")
            break

        if not email_text:
            print("Please enter a non-empty email.")
            continue

        predict_email(email_text)
