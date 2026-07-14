import os
import argparse
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_pipeline():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    fe_path = os.path.join(MODEL_DIR, "feature_engineer.pkl")
    prep_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

    if not all(os.path.exists(p) for p in [model_path, fe_path, prep_path]):
        raise FileNotFoundError(
            "Trained model files not found in models/. Run `python train.py` first."
        )

    model = joblib.load(model_path)
    feature_engineer = joblib.load(fe_path)
    preprocessor = joblib.load(prep_path)
    return model, feature_engineer, preprocessor


def predict_text(text, model, feature_engineer, preprocessor):
    cleaned = [preprocessor.clean_text(text)]
    raw = [text]
    X = feature_engineer.transform(cleaned, raw)

    pred = model.predict(X)[0]
    label = "REAL" if pred == 1 else "FAKE"

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = probs[pred]

    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict REAL vs FAKE for a news article.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw article text to classify.")
    group.add_argument("--file", type=str, help="Path to a .txt file containing the article.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    model, feature_engineer, preprocessor = load_pipeline()
    label, confidence = predict_text(text, model, feature_engineer, preprocessor)

    print("\n" + "=" * 50)
    print(f"Prediction: {label}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
