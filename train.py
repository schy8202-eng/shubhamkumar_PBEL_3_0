import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import get_model_zoo, train_and_compare, tune_best_model
from src.evaluate import (
    plot_confusion_matrix, plot_model_comparison, plot_roc_curves,
    top_predictive_words, full_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sample_dataset():
    path = os.path.join(DATA_DIR, "news_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python data/create_sample_data.py` first."
        )
    df = pd.read_csv(path)
    return df


def load_kaggle_dataset():
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        raise FileNotFoundError(
            "Fake.csv / True.csv not found in data/. Download the Kaggle "
            "'Fake and Real News Dataset' and place both files in data/."
        )
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"
    df = pd.concat([fake_df, true_df], ignore_index=True)
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    df = df[["text", "label"]].dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main(use_kaggle=False, sample_size=None):
    print("=" * 60)
    print("FAKE NEWS DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Loading dataset...")
    df = load_kaggle_dataset() if use_kaggle else load_sample_dataset()
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Loaded {len(df)} articles. Label distribution:\n{df['label'].value_counts()}")

    # 2. Preprocess text
    print("\n[2/7] Preprocessing text (cleaning, stopwords, lemmatization)...")
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.clean_series(df["text"])
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"  Done. {len(df)} articles remain after cleaning.")

    # 3. Encode labels
    df["label_encoded"] = (df["label"].str.upper() == "REAL").astype(int)  # REAL=1, FAKE=0

    # 4. Train/test split
    print("\n[3/7] Splitting train/test sets (80/20, stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[["text", "clean_text"]], df["label_encoded"],
        test_size=0.2, random_state=42, stratify=df["label_encoded"]
    )

    # 5. Feature engineering
    print("\n[4/7] Extracting TF-IDF + handcrafted features...")
    fe = FeatureEngineer(max_features=5000, ngram_range=(1, 2))
    X_train = fe.fit_transform(X_train_raw["clean_text"], X_train_raw["text"])
    X_test = fe.transform(X_test_raw["clean_text"], X_test_raw["text"])
    print(f"  Feature matrix shape: train={X_train.shape}, test={X_test.shape}")

    # 6. Balance classes with SMOTE (on training data only)
    print("\n[5/7] Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  Balanced training set shape: {X_train_bal.shape}, "
          f"class distribution: {pd.Series(y_train_bal).value_counts().to_dict()}")

    # 7. Train & compare models
    print("\n[6/7] Training and comparing models...")
    models = get_model_zoo()
    results_df, fitted_models = train_and_compare(models, X_train_bal, y_train_bal, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    print(f"\n🏆 Best model: {best_model_name}")

    # Detailed report + confusion matrix for best model
    y_pred_best = best_model.predict(X_test)
    full_report(y_test, y_pred_best, best_model_name)
    plot_confusion_matrix(y_test, y_pred_best, best_model_name, OUTPUT_DIR)
    plot_model_comparison(results_df, OUTPUT_DIR)
    plot_roc_curves(fitted_models, X_test, y_test, OUTPUT_DIR)

    # Feature importance (top predictive words) if the underlying model supports it
    try:
        raw_best = best_model.estimator if hasattr(best_model, "estimator") else best_model
        top_fake, top_real = top_predictive_words(raw_best, fe.get_feature_names())
        if top_fake:
            print("\nTop words/features pushing toward FAKE:")
            for word, score in top_fake[:10]:
                print(f"    {word}: {score:.4f}")
        if top_real:
            print("\nTop words/features pushing toward REAL:")
            for word, score in top_real[:10]:
                print(f"    {word}: {score:.4f}")
    except Exception as e:
        print(f"  (Feature importance not available for this model: {e})")

    # 8. Optional hyperparameter tuning of best model
    print(f"\n[7/7] Attempting hyperparameter tuning for {best_model_name}...")
    tuned_model = tune_best_model(best_model_name, X_train_bal, y_train_bal)
    final_model = tuned_model if tuned_model is not None else best_model

    # Save everything needed for inference
    joblib.dump(final_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(fe, os.path.join(MODEL_DIR, "feature_engineer.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
    with open(os.path.join(MODEL_DIR, "model_name.txt"), "w") as f:
        f.write(best_model_name)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)

    print(f"\n Training complete. Best model ({best_model_name}) and pipeline saved to {MODEL_DIR}/")
    print(f" Evaluation plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    parser.add_argument("--kaggle", action="store_true", help="Use real Kaggle dataset instead of synthetic sample.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optionally subsample the dataset for faster runs.")
    args = parser.parse_args()
    main(use_kaggle=args.kaggle, sample_size=args.sample_size)
