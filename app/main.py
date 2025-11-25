import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

@st.cache_resource
def load_artifacts(models_dir: Path):
    preproc_path = models_dir / "preprocessing_pipeline.joblib"
    model_path = models_dir / "linear_model.joblib"
    meta_path = models_dir / "metadata.json"

    if not preproc_path.exists() or not model_path.exists():
        return None, None, None

    preproc = joblib.load(preproc_path)
    model = joblib.load(model_path)

    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as fh:
            meta = json.load(fh)

    return preproc, model, meta


def main():
    st.set_page_config(page_title="House Price Predictor (Simple Mode)", layout="wide")
    st.title("🏡 House Price Predictor — Simple Input Mode")

    # Path setup
    repo_root = Path(__file__).resolve().parents[1]
    print(repo_root)
    models_dir = repo_root / "model"
    # /Users/vinaysharma/SalesForeCastingMLProject/model
    # Load model, preprocessor, metadata
    preproc, model, meta = load_artifacts(models_dir)

    if preproc is None or model is None:
        st.error("Model not found! Train the model first.")
        return

    if not meta:
        st.error("Metadata missing! Re-train with src/train_pipeline.py.")
        return

    st.write("### Enter only the most important features")
    st.write("Remaining 80+ features will be automatically filled with default values.")

    # Important features only
    important_features = {
        "OverallQual": ("int", 5),
        "GrLivArea": ("int", 1500),
        "GarageCars": ("int", 2),
        "GarageArea": ("int", 400),
        "TotalBsmtSF": ("int", 1000),
        "YearBuilt": ("int", 2000),
        "YearRemodAdd": ("int", 2010),
        "FullBath": ("int", 2),
        "BedroomAbvGr": ("int", 3),
        "KitchenQual": ("cat", meta["categories_sample"]["KitchenQual"]),
        "Neighborhood": ("cat", meta["categories_sample"]["Neighborhood"]),
        "LotArea": ("int", 7000),
    }

    # Real full list from metadata
    all_features = meta["feature_names"]
    dtypes = meta["dtypes"]
    categories = meta["categories_sample"]

    # UI input form
    with st.form("prediction_form"):
        inputs = {}

        col1, col2, col3 = st.columns(3)

        for i, (feat, (ftype, default)) in enumerate(important_features.items()):
            col = [col1, col2, col3][i % 3]

            if ftype == "int":
                inputs[feat] = col.number_input(feat, value=default)
            else:
                inputs[feat] = col.selectbox(feat, [""] + default)

        submitted = st.form_submit_button("Predict Sale Price")

    if not submitted:
        return

    # Build full input row with defaults
    final_input = {}

    for feat in all_features:
        if feat in inputs:
            final_input[feat] = inputs[feat]  # user input
        else:
            # Auto-fill other features using defaults
            if dtypes[feat] in ["int64", "float64"]:
                final_input[feat] = 0  # safe numeric default
            else:
                # pick most frequent category if available
                if feat in categories:
                    final_input[feat] = categories[feat][0]
                else:
                    final_input[feat] = ""

    df_row = pd.DataFrame([final_input])

    # Prediction
    try:
        X_t = preproc.transform(df_row)
        pred_log = model.predict(X_t)[0]
        pred = np.expm1(pred_log)

        st.success(f"💰 Predicted Sale Price: **${pred:,.2f}**")

    except Exception as e:
        st.error("Prediction failed! Check inputs or retrain.")
        st.exception(e)


if __name__ == "__main__":
    main()
