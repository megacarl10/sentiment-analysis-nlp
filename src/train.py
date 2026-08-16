import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)

from preprocessing import preprocess_text


# load dataset
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_twitter.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Preprocess text
print("\nPreprocessing text...")

df["clean_text"] = df["text"].apply(preprocess_text)

# Remove empty rows after preprocessing
df = df[df["clean_text"].str.strip() != ""]
print("Preprocessing complete.")

# train/test split

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

#TF-IDF

print("\nCreating TF-IDF features...")

vectorizer=TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf= vectorizer.fit_transform(X_train)
X_test_tfidf= vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

#save vectorizer
joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

#define models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Naive Bayes": MultinomialNB(),

    "Linear SVM": CalibratedClassifierCV(
        LinearSVC(C=1.0),
        cv=3
    )
}

#train and evalutate
results={}
probability_predictions={}

for name,model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {name}...")
    print("=" * 50)

    model.fit(X_train_tfidf, y_train)

    #predictions
    y_pred = model.predict(X_test_tfidf)

    probabilities = model.predict_proba(X_test_tfidf)
    probability_predictions[name] = probabilities

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_tfidf)

        print("\nProbability shape:", probabilities.shape)

        print("First prediction probablities:")
        print(probabilities[0])

        print("Probability sum:")
        print(probabilities[0].sum())

    #accuracy
    accuracy = accuracy_score(y_test,y_pred)
    results[name] = accuracy
    print(f"\n{name}Accuracy:, {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test,y_pred))

    #save model
    filename=name.lower().replace(" ", "-") + ".joblib"

    joblib.dump(
        model,
        MODEL_DIR / filename
    )
    print(f"\nModel saved: models/{filename}")

#ensemble prediction

print("\n\n" + "=" * 50)
print("ENSEMBLE MODEL")
print("=" * 50)

# Average probabilities from all models
ensemble_probabilities = (
    0.2 * probability_predictions["Logistic Regression"]
    + 0.1 * probability_predictions["Naive Bayes"]
    + 0.7 * probability_predictions["Linear SVM"]
) / 3


# Convert probabilities to class predictions
class_names = models["Logistic Regression"].classes_

ensemble_predictions = class_names[
    ensemble_probabilities.argmax(axis=1)
]

# Evaluate ensemble
ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_predictions
)

print(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}")

print("\nEnsemble Classification Report:")
print(
    classification_report(
        y_test,
        ensemble_predictions
    )
)

print("\nEnsemble Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        ensemble_predictions
    )
)

#model comparision
print("\n\n" + "=" * 50)
print("MODEL COMPARISION")
print("=" * 50)

for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

print(f"Ensemble: {ensemble_accuracy:.4f}")