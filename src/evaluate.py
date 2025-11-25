"""Evaluate saved pipeline/model against a CSV containing the ground-truth SalePrice.

Generates metrics (RMSE, MAE, R2, MAPE) and saves a small report plus residuals CSV in the models directory.

Usage:
    python src/evaluate.py --data data/raw_data/train.csv --models models
"""
# python3 src/evaluate.py --data data/raw_data/train.csv --models models --out models
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_pipeline_or_parts(models_dir: Path):
    """Try to load a saved end-to-end pipeline, falling back to preprocessor + model."""
    full_path = models_dir / 'full_pipeline.joblib'
    preproc_path = models_dir / 'preprocessing_pipeline.joblib'
    model_path = models_dir / 'linear_model.joblib'

    if full_path.exists():
        pipe = joblib.load(full_path)
        return pipe, True

    if preproc_path.exists() and model_path.exists():
        preproc = joblib.load(preproc_path)
        model = joblib.load(model_path)
        return (preproc, model), False

    raise FileNotFoundError('No pipeline or model artifacts found in models dir')


def predict_with_artifacts(artifacts, is_full_pipeline, X: pd.DataFrame):
    if is_full_pipeline:
        pipe = artifacts
        preds_log = pipe.predict(X)
    else:
        preproc, model = artifacts
        X_t = preproc.transform(X)
        preds_log = model.predict(X_t)

    # training used np.log1p on the target, so inverse with expm1
    preds = np.expm1(preds_log)
    return preds


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # avoid division by zero in MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(
            np.abs((y_true - y_pred) /
            np.where(y_true == 0, np.nan, y_true))
        ) * 100

    return dict(rmse=float(rmse), mae=float(mae), r2=float(r2), mape=float(mape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--target', type=str, default='SalePrice')
    parser.add_argument('--out', type=Path, default=None,
                        help='optional output dir to save report (defaults to models dir)')
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise RuntimeError(f'Target column {args.target} not present in provided CSV')

    y = df[args.target].values
    artifacts, is_full = load_pipeline_or_parts(args.models)

    # Attempt to select features using metadata
    meta_path = args.models / 'metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r') as fh:
            meta = json.load(fh)
        feature_names = meta.get('feature_names', None)
    else:
        feature_names = None

    if feature_names:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise RuntimeError(
                f'Missing feature columns required for prediction: {missing[:10]}'
            )
        X = df[feature_names]
    else:
        # Use all columns except target
        X = df.drop(columns=[args.target])

    preds = predict_with_artifacts(artifacts, is_full, X)
    metrics = compute_metrics(y, preds)

    out_dir = args.out or args.models
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    report_path = out_dir / 'eval_metrics.json'
    with open(report_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # Save residuals
    residuals = df.copy()
    residuals['prediction'] = preds
    residuals['residual'] = residuals[args.target] - residuals['prediction']
    residuals_path = out_dir / 'eval_residuals.csv'
    residuals.to_csv(residuals_path, index=False)

    print('Evaluation results:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    print('Saved report to', report_path)
    print('Saved residuals to', residuals_path)


if __name__ == '__main__':
    main()
