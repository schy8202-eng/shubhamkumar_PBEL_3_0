import os
import joblib
import streamlit as st
import pandas as pd

from predict import load_pipeline, predict_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("Fake News Detection using Machine Learning")
st.markdown(
    "This app uses a machine learning model to predict whether a news article is **REAL** or **FAKE** based on its text content. "
)

model_files_exist = all(
    os.path.exists(os.path.join(MODEL_DIR, f))
    for f in ["best_model.pkl", "feature_engineer.pkl", "preprocessor.pkl"]
)

if not model_files_exist:
    st.error(
        " No trained model found. Run `python train.py` in the project "
        "folder first to train and save a model."
    )
    st.stop()


@st.cache_resource
def get_pipeline():
    return load_pipeline()


model, feature_engineer, preprocessor = get_pipeline()

model_name_path = os.path.join(MODEL_DIR, "model_name.txt")
model_name = "ML Model"
if os.path.exists(model_name_path):
    with open(model_name_path) as f:
        model_name = f.read().strip()

st.caption(f"Currently serving: **{model_name}**")

text_input = st.text_area(
    "Article text",
    height=220,
    placeholder="Paste the full text of a news article here...",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button(" Analyze", type="primary", use_container_width=True)

if predict_clicked:
    if not text_input.strip():
        st.warning("Please paste some article text first.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict_text(text_input, model, feature_engineer, preprocessor)

        if label == "REAL":
            st.success(f" Prediction: **REAL**")
        else:
            st.error(f" Prediction: **FAKE**")

        if confidence is not None:
            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence:.1%}")

        with st.expander("Why might this be flagged this way?"):
            st.markdown(
                "This AI model analyzes the news article using:" \
                "Important keywords and phrases (TF-IDF)" \
                "Writing Style and sentence patterns " \
                "**NOTE:**" \
                "This prediction is based on machine learning patterns.It does not verify facts from the internet.Always cross-check important news with trusted news sources." 

                
            )

st.divider()

# Show model comparison results if available
results_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
if os.path.exists(results_path):
    with st.expander(" View model comparison results"):
        df = pd.read_csv(results_path)
        st.dataframe(df, use_container_width=True)

st.caption(
    " Developed by Shubham Kumar | AI-Based Fake News Detection System"
       "Using Machine Learning." \
       "Build with python, Streamlit and scikit-learn."
       
)
