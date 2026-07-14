# Fake News Detection System

An end-to-end machine learning system that classifies news articles as **REAL** or **FAKE** using NLP feature engineering and multiple ML models, with a Streamlit web app for live predictions.

## Features

- **Robust text preprocessing** — HTML/URL stripping, punctuation & noise removal, stopword filtering, lemmatization
- **Rich feature engineering**:
  - TF-IDF (unigrams + bigrams, 5000 features)
  - Handcrafted linguistic features: text length, word count, punctuation ratio, uppercase ratio, exclamation/question mark counts, average word length, stopword ratio
- **6 ML models trained & compared**:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM
  - Random Forest
  - Gradient Boosting (XGBoost)
  - Passive Aggressive Classifier
- **Class imbalance handling** via SMOTE
- **Hyperparameter tuning** with GridSearchCV on the best-performing model
- **Full evaluation suite** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves, feature importance / most predictive words
- **Model persistence** (joblib) — train once, reuse anywhere
- **Streamlit web app** — paste any article and get an instant prediction with confidence score
- **CLI prediction tool** for scripted/batch use

## 📁 Project Structure

```
fake_news_detection/
├── data/
│   └── create_sample_data.py    # generates a synthetic dataset for demo/testing
├── src/
│   ├── preprocessing.py         # text cleaning class
│   ├── feature_engineering.py   # TF-IDF + handcrafted features
│   ├── models.py                # model definitions, training, comparison
│   └── evaluate.py              # metrics, plots, reports
├── models/                      # saved trained models (.pkl) after training
├── outputs/                     # evaluation plots & reports
├── train.py                     # main training pipeline (run this first)
├── predict.py                   # CLI prediction script
├── app.py                       # Streamlit web app
├── requirements.txt
└── README.md
```

## 🏁 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a dataset

**Option A — Use the real Kaggle dataset (recommended for real results):**
Download the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — it gives you `Fake.csv` and `True.csv`. Place both files in the `data/` folder.

**Option B — Generate a synthetic sample dataset (for quickly testing the pipeline works):**

```bash
python data/create_sample_data.py
```

This creates `data/news_dataset.csv` with a small labeled synthetic dataset so you can run the whole pipeline end-to-end immediately. Swap it for the real Kaggle data when you want production-quality accuracy.

### 3. Train all models

```bash
python train.py
```

This will:

- Load and clean the data
- Extract TF-IDF + handcrafted features
- Balance classes with SMOTE
- Train all 6 models and print a comparison table
- Tune the best model with GridSearchCV
- Save the best model + vectorizer to `models/`
- Save evaluation plots to `outputs/`

### 4. Predict on new text (CLI)

```bash
python predict.py --text "Breaking: Scientists discover shocking new evidence..."
```

### 5. Launch the web app

```bash
streamlit run app.py
```

Paste any article text into the browser UI and get an instant REAL/FAKE prediction with a confidence score.

## How It Works

1. **Preprocessing** — lowercase, strip URLs/HTML/special characters, tokenize, remove stopwords, lemmatize
2. **Feature extraction** — combines sparse TF-IDF vectors with dense handcrafted stylistic features (fake news tends to use more exclamation marks, more uppercase words, different sentence structure, etc.)
3. **Model training** — trains multiple algorithms on the same features so you can see which approach works best for your data, using 5-fold stratified cross-validation
4. **Evaluation** — confusion matrices, ROC-AUC, precision/recall/F1 per model, plus a look at which words are most predictive of fake vs real
5. **Deployment** — the best model is serialized and served through a Streamlit interface

## 🔧 Extending This Project

- Swap TF-IDF for word embeddings (Word2Vec, GloVe) or transformer embeddings (BERT/DistilBERT via `sentence-transformers`)
- Add an LSTM/GRU deep learning model with Keras for sequence-level understanding
- Add source credibility / metadata features (publisher, author, date) if available in your dataset
- Deploy the Streamlit app to Streamlit Community Cloud or wrap `predict.py` in a Flask/FastAPI REST endpoint
- Add explainability with LIME or SHAP to show _why_ an article was flagged as fake

## Expected Performance

On the real Kaggle dataset, TF-IDF + Logistic Regression / SVM / XGBoost typically reach **95–99% accuracy** — this dataset is a well-known benchmark. The synthetic sample data included here is only meant to prove the pipeline runs; it is too small and simple to produce meaningful accuracy numbers.
