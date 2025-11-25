from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame, ignore_cols: List[str] = None) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    """Build a ColumnTransformer for a dataframe.

    Returns (preprocessor, metadata) where metadata contains lists of numeric and categorical columns
    and a small sample of category values to help UI rendering.
    """
    if ignore_cols is None:
        ignore_cols = []

    df_work = df.drop(columns=[c for c in ignore_cols if c in df.columns], errors='ignore')

    numeric_cols = df_work.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_work.select_dtypes(include=['object', 'category']).columns.tolist()

    # numeric pipeline: impute median, scale
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical pipeline: impute constant and one-hot
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        # sklearn >=1.2 renamed `sparse` to `sparse_output`
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    # [('num', numeric_pipeline, numeric_cols),]
    if numeric_cols:
        transformers.append(('num', numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # Collect small metadata for UI
    metadata = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'categories_sample': {c: df[c].dropna().unique()[:50].tolist() for c in categorical_cols}
    }

    return preprocessor, metadata
