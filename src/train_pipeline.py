"""Train pipeline for House Prices dataset (target: SalePrice).

Saves the preprocessing pipeline, the trained model, and a small metadata json that
the Streamlit app uses to render inputs.

Usage:
    python src/train_pipeline.py --data data/raw_data/train.csv --out models
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data_preprocessing import build_preprocessor


def train(data_path: Path, out_dir: Path, test_size=0.15, random_state=42):
    df = pd.read_csv(data_path)
    if 'SalePrice' not in df.columns:
        raise RuntimeError('Input CSV must contain a `SalePrice` column as target')

    # keep Id if present but don't use it as a feature
    ignore_cols = ['Id'] if 'Id' in df.columns else []

    y = df['SalePrice'].copy()
    X = df.drop(columns=['SalePrice'] + [c for c in ignore_cols if c in df.columns], errors='ignore')

    # log-transform target to stabilise variance
    y_trans = np.log1p(y.values)

    preprocessor, metadata = build_preprocessor(X)

    model = LinearRegression()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y_trans, test_size=test_size, random_state=random_state)

    print('Fitting pipeline...')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    # Some sklearn versions accept `squared=False`; to be maximally compatible compute RMSE explicitly
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'Validation RMSE (log-target): {rmse:.4f}')

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the full pipeline and a model-only artifact for convenience
    preprocessing_path = out_dir / 'preprocessing_pipeline.joblib'
    model_path = out_dir / 'linear_model.joblib'
    pipeline_path = out_dir / 'full_pipeline.joblib'

    # Save preprocessor alone
    joblib.dump(preprocessor, preprocessing_path)
    # Save only the fitted sklearn model (the RandomForest inside pipeline)
    joblib.dump(pipeline.named_steps['model'], model_path)
    # Save the end-to-end pipeline too (useful for batch predictions)
    joblib.dump(pipeline, pipeline_path)

    # metadata: feature names (raw), and small samples of categorical values
    metadata['feature_names'] = X.columns.tolist()
    metadata['dtypes'] = {c: str(X[c].dtype) for c in X.columns}

    # shrink categories_sample to lists
    metadata_path = out_dir / 'metadata.json'
    with open(metadata_path, 'w') as fh:
        json.dump(metadata, fh, indent=2)

    print('Saved artifacts:')
    print(f'  {preprocessing_path}')
    print(f'  {model_path}')
    print(f'  {pipeline_path}')
    print(f'  {metadata_path}')

train('data/raw_data/train.csv', Path('model'))
# # Command line Interface 
# def cli():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=Path, default=Path('data/raw_data/train.csv'))
#     parser.add_argument('--out', type=Path, default=Path('models'))
#     args = parser.parse_args()
#     # print(args.out)
#     train(args.data, args.out)


# if __name__ == '__main__':
#     cli()
