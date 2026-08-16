from pathlib import Path

import joblib

from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocessing import preprocess_text

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

#paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

#load models
vectorizer=joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)
logistic_model = joblib.load(
    MODEL_DIR / "logistic-regression.joblib"
)
naive_bayes_model = joblib.load(
    MODEL_DIR / "naive-bayes.joblib"
)
svm_model = joblib.load(
    MODEL_DIR / "linear-svm.joblib"
)

#fastapi
app=FastAPI(
    title="Sentiment Analysis API",
    description="NLP sentiment classification using TF-IDF and machine learning",
    version="1.0"
)
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

#request schema
class TextRequest(BaseModel):
    text:str

#home endpoint
@app.get("/")
def home():
    return{
        "text": "Sentiment Analysis API is running"
    }

@app.get("/app")
def frontend():
    return FileResponse("static/index.html")

#prediction endpoint
@app.post("/predict")
def predict(request: TextRequest):

    #preprocess input
    clean_text = preprocess_text(request.text)

    #convert text to tf-idf
    text_vector=vectorizer.transform([clean_text])

    #individual prediction
    lr_prob = logistic_model.predict_proba(text_vector)[0]
    nb_prob = naive_bayes_model.predict_proba(text_vector)[0]
    svm_prob = svm_model.predict_proba(text_vector)[0]

    #svm prediction
    svm_prediction=svm_model.predict(text_vector)[0]


    #weighted ensemble
    ensemble_prob = (
        0.2 * lr_prob + 0.1 * nb_prob + 0.7 * svm_prob
    )
    classes= svm_model.classes_

    ensemble_prediction = classes[
        ensemble_prob.argmax()
    ]

    #probability dictionairies

    svm_probabilities = {
        class_name: round(float(probability), 4)
        for class_name, probability
        in zip(classes, svm_prob)
    }
    ensemble_probabilities = {
        class_name: round(float(probability), 4)
        for class_name, probability
        in zip(classes, ensemble_prob)
    }

    return{
        "text": request.text, 
        "svm_prediction": svm_prediction,
        "svm_probabilities": svm_probabilities,
        "ensemble_prediction": ensemble_prediction,
        "ensemble_probabilities": ensemble_probabilities
    }