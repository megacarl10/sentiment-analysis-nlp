# Sentiment Analysis using NLP and Machine Learning

An end-to-end sentiment analysis application that classifies text into **Positive, Negative, or Neutral** sentiment using TF-IDF feature extraction and multiple machine learning models.

The project compares Logistic Regression, Multinomial Naive Bayes, and Calibrated Linear SVM, and implements a weighted ensemble for comparison. A FastAPI backend provides real-time sentiment predictions through a REST API and a simple web interface.

---

## Features

- Text preprocessing using NLTK
- TF-IDF feature extraction with unigram and bigram features
- Multiple machine learning classifiers
- Calibrated Linear SVM for probability-based predictions
- Weighted ensemble prediction
- Model performance comparison
- Confusion matrix and classification reports
- REST API using FastAPI
- Interactive Swagger API documentation
- Simple web interface for real-time predictions
- Serialized models using Joblib

---

## Machine Learning Models

The project evaluates three classifiers:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Calibrated Linear SVM

The models use the same:

- Dataset
- Preprocessing pipeline
- Train/test split
- TF-IDF representation

This allows a fair comparison between classifiers.

---

## Dataset

The project uses a Twitter sentiment dataset containing approximately **74,000 tweets** across multiple sentiment categories.

After removing missing values and irrelevant labels, the dataset contains:

- **57,297 samples**
- **3 sentiment classes**

| Sentiment | Samples |
|-----------|---------|
| Negative | 21,171 |
| Positive | 19,078 |
| Neutral | 17,048 |

The data is split using a stratified 80/20 train-test split.

### Dataset Split

- Training samples: **45,572**
- Testing samples: **11,393**

---

## Text Preprocessing

The text preprocessing pipeline includes:

- Lowercasing
- Tokenization
- Stop-word removal
- Lemmatization
- Removal of unnecessary characters

The cleaned text is then converted into numerical features using TF-IDF.

---

## TF-IDF Configuration

The vectorizer uses:

```text
max_features = 20,000
ngram_range = (1, 2)
min_df = 2
sublinear_tf = True