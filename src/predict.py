from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import json
from typing import Dict, Any


def load_artifacts(models_dir: Path):
    preprocessing_path = models_dir / 'preprocessing_pipeline.joblib'
    model_path = models_dir / 'rf_model.joblib'
    metadata_path = models_dir / 'metadata.json'

    if not preprocessing_path.exists() or not model_path.exists():
        raise FileNotFoundError('Pipeline/model artifacts not found in models dir')

    preprocessor = joblib.load(preprocessing_path)
    model = joblib.load(model_path)
    meta = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as fh:
            meta = json.load(fh)

    return preprocessor, model, meta


def predict_from_df(df: pd.DataFrame, models_dir: Path) -> pd.Series:
    preprocessor, model, meta = load_artifacts(models_dir)
    # keep same columns as metadata if present
    if meta and 'feature_names' in meta:
        X = df[meta['feature_names']]
    else:
        X = df

    X_trans = preprocessor.transform(X)
    preds_log = model.predict(X_trans)
    preds = np.expm1(preds_log)
    return pd.Series(preds, index=df.index)


def predict_single(row: Dict[str, Any], models_dir: Path):
    df = pd.DataFrame([row])
    return predict_from_df(df, models_dir).iloc[0]


if __name__ == '__main__':
    # quick CLI to predict a CSV
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models'))
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    preds = predict_from_df(df, args.models)
    out = args.input.with_name(args.input.stem + '_preds.csv')
    df_out = df.copy()
    df_out['prediction'] = preds
    df_out.to_csv(out, index=False)
    print('Saved predictions to', out)
