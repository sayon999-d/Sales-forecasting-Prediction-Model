from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4
from urllib.parse import urlparse
from zipfile import ZipFile

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(".")
MODEL_DIR = APP_DIR / "models"
ARTIFACT_DIR = APP_DIR / "artifacts"
KAGGLE_CACHE_DIR = APP_DIR / "data" / "kaggle_cache"
MODEL_BUNDLE_PATH = MODEL_DIR / "streamlit_model_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "streamlit_metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "streamlit_sample_predictions.csv"
PROVIDED_KAGGLE_SOURCES = [
    {
        "label": "Sales Forecasting",
        "url": "https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting",
        "kind": "dataset",
        "note": "Generic sales dataset.",
    },
    {
        "label": "Walmart Sales Forecast",
        "url": "https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast",
        "kind": "dataset",
        "note": "Walmart-focused forecasting dataset.",
    },
    {
        "label": "Retail Sales Forecasting",
        "url": "https://www.kaggle.com/datasets/tevecsystems/retail-sales-forecasting",
        "kind": "dataset",
        "note": "Retail sales forecasting dataset.",
    },
]


@st.cache_data(show_spinner=False)
def load_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_preview(path: str, nrows: int = 200) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner=False)
def get_csv_metadata(path: str) -> dict[str, Any]:
    preview = load_csv_preview(path, nrows=200)
    return {
        "columns": preview.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in preview.dtypes.items()},
        "preview_rows": len(preview),
    }


@st.cache_data(show_spinner=False)
def get_common_columns(paths: tuple[str, ...]) -> list[str]:
    if not paths:
        return []

    shared_columns = set(get_csv_metadata(paths[0])["columns"])
    first_order = get_csv_metadata(paths[0])["columns"]

    for path in paths[1:]:
        shared_columns &= set(get_csv_metadata(path)["columns"])

    return [column for column in first_order if column in shared_columns]


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_kaggle_source(source: str) -> tuple[str, str]:
    if source.startswith("http://") or source.startswith("https://"):
        parsed = urlparse(source)
        path_parts = [part for part in parsed.path.split("/") if part]

        if len(path_parts) >= 3 and path_parts[0] == "datasets":
            return "dataset", f"{path_parts[1]}/{path_parts[2]}"

        if len(path_parts) >= 2 and path_parts[0] in {"competitions", "c"}:
            return "competition", path_parts[1]

        raise ValueError(f"Unsupported Kaggle URL format: {source}")

    if "/" in source:
        return "dataset", source.strip("/")

    return "competition", source.strip("/")


def configure_kaggle_auth() -> None:
    load_env_file(Path(".env"))

    workspace_kaggle_dir = Path(".kaggle")
    if workspace_kaggle_dir.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(workspace_kaggle_dir.resolve())
        return

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return

    raise RuntimeError(
        "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in .env."
    )


def sanitize_identifier(identifier: str) -> str:
    return identifier.replace("/", "__")


def get_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The kaggle package is not installed. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    configure_kaggle_auth()
    api = KaggleApi()
    api.authenticate()
    return api


def test_kaggle_connection() -> dict[str, Any]:
    api = get_kaggle_api()
    probe_summary = {
        "authenticated": True,
        "username": os.environ.get("KAGGLE_USERNAME", ""),
        "probe": "authentication_only",
    }

    list_method = None
    for candidate in ("dataset_list", "datasets_list"):
        if hasattr(api, candidate):
            list_method = getattr(api, candidate)
            break

    if list_method is not None:
        try:
            probe_result = list_method(search="sales")
            probe_count = len(probe_result) if hasattr(probe_result, "__len__") else None
            probe_summary["probe"] = "dataset_list"
            probe_summary["sample_result_count"] = probe_count
        except TypeError:
            probe_result = list_method()
            probe_count = len(probe_result) if hasattr(probe_result, "__len__") else None
            probe_summary["probe"] = "dataset_list"
            probe_summary["sample_result_count"] = probe_count

    return probe_summary


def extract_competition_zip(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected downloaded archive not found: {zip_path}")

    with ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def build_fetch_result(
    source: str,
    source_kind: str,
    identifier: str,
    target_dir: Path,
    from_cache: bool,
) -> dict[str, Any]:
    csv_files = sorted(path for path in target_dir.rglob("*.csv") if path.is_file())
    return {
        "source": source,
        "source_kind": source_kind,
        "identifier": identifier,
        "directory": str(target_dir),
        "csv_files": [str(path) for path in csv_files],
        "from_cache": from_cache,
    }


def source_cache_status(source: str) -> dict[str, Any]:
    source_kind, identifier = parse_kaggle_source(source)
    target_dir = KAGGLE_CACHE_DIR / sanitize_identifier(identifier)
    csv_files = sorted(target_dir.rglob("*.csv")) if target_dir.exists() else []
    return {
        "source": source,
        "source_kind": source_kind,
        "identifier": identifier,
        "directory": str(target_dir),
        "cached": bool(csv_files),
        "csv_count": len(csv_files),
    }


def raise_friendly_kaggle_error(
    source_kind: str,
    identifier: str,
    exc: Exception,
) -> None:
    error_text = str(exc)
    if "401" in error_text or "Unauthorized" in error_text:
        if source_kind == "competition":
            raise RuntimeError(
                f"Kaggle competition access failed for '{identifier}'. "
                "Open the competition page in Kaggle, accept the competition rules, "
                "then try again. If you already accepted them, verify that "
                "KAGGLE_USERNAME and KAGGLE_KEY in .env are correct and restart Streamlit."
            ) from exc
        raise RuntimeError(
            f"Kaggle authentication failed for '{identifier}'. "
            "Check KAGGLE_USERNAME and KAGGLE_KEY in .env, then restart Streamlit."
        ) from exc

    raise RuntimeError(
        f"Kaggle fetch failed for '{identifier}': {error_text}"
    ) from exc


def fetch_kaggle_source_uncached(source: str) -> dict[str, Any]:
    source_kind, identifier = parse_kaggle_source(source)
    target_dir = KAGGLE_CACHE_DIR / sanitize_identifier(identifier)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_csv_files = sorted(path for path in target_dir.rglob("*.csv") if path.is_file())
    if existing_csv_files:
        return build_fetch_result(
            source=source,
            source_kind=source_kind,
            identifier=identifier,
            target_dir=target_dir,
            from_cache=True,
        )

    api = get_kaggle_api()

    try:
        if source_kind == "dataset":
            api.dataset_download_files(
                identifier,
                path=str(target_dir),
                unzip=True,
                quiet=True,
            )
        else:
            api.competition_download_files(
                identifier,
                path=str(target_dir),
                quiet=True,
            )
            extract_competition_zip(target_dir / f"{identifier}.zip", target_dir)
    except Exception as exc:
        raise_friendly_kaggle_error(source_kind, identifier, exc)

    return build_fetch_result(
        source=source,
        source_kind=source_kind,
        identifier=identifier,
        target_dir=target_dir,
        from_cache=False,
    )


@st.cache_data(show_spinner=False)
def fetch_kaggle_source(source: str) -> dict[str, Any]:
    return fetch_kaggle_source_uncached(source)


def fetch_kaggle_sources_parallel(sources: list[str], max_workers: int = 3) -> list[dict[str, Any]]:
    if not sources:
        return []

    results_by_source: dict[str, dict[str, Any]] = {}
    worker_count = min(max_workers, len(sources))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_source = {
            executor.submit(fetch_kaggle_source_uncached, source): source for source in sources
        }
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            results_by_source[source] = future.result()

    return [results_by_source[source] for source in sources]


def infer_datetime_columns(df: pd.DataFrame) -> list[str]:
    datetime_columns: list[str] = []
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_columns.append(column)
            continue

        if df[column].dtype != "object":
            continue

        sample = df[column].dropna().head(200)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            datetime_columns.append(column)

    return datetime_columns


def enrich_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for column in infer_datetime_columns(enriched):
        parsed = pd.to_datetime(enriched[column], errors="coerce")
        enriched[f"{column}_year"] = parsed.dt.year
        enriched[f"{column}_month"] = parsed.dt.month
        enriched[f"{column}_day"] = parsed.dt.day
        enriched[f"{column}_dayofweek"] = parsed.dt.dayofweek
        enriched[f"{column}_quarter"] = parsed.dt.quarter
        enriched = enriched.drop(columns=[column])
    return enriched


def combine_dataframes(
    frames: list[pd.DataFrame],
    source_labels: list[str],
    target_column: str,
    merge_strategy: str,
) -> tuple[pd.DataFrame, list[str]]:
    prepared_frames: list[pd.DataFrame] = []

    for label, frame in zip(source_labels, frames, strict=True):
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' missing in {label}.")

        tagged = frame.copy()
        tagged["source_file"] = label
        prepared_frames.append(tagged)

    if merge_strategy == "common_columns":
        common_columns = set(prepared_frames[0].columns)
        for frame in prepared_frames[1:]:
            common_columns &= set(frame.columns)

        if target_column not in common_columns:
            raise ValueError(
                "The target column is not shared across all selected CSV files."
            )

        ordered_columns = [
            column for column in prepared_frames[0].columns if column in common_columns
        ]
        combined = pd.concat(
            [frame[ordered_columns] for frame in prepared_frames],
            ignore_index=True,
        )
        return combined, ordered_columns

    combined = pd.concat(prepared_frames, ignore_index=True, sort=False)
    ordered_columns = combined.columns.tolist()
    return combined, ordered_columns


def create_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in x.columns if column not in numeric_features
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", create_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_models(
    preprocessor: ColumnTransformer,
    random_state: int,
    training_profile: str,
) -> dict[str, Pipeline]:
    profile_settings = {
        "fast": {
            "rf_estimators": 60,
            "et_estimators": 90,
            "models": ["linear_regression", "extra_trees"],
        },
        "balanced": {
            "rf_estimators": 140,
            "et_estimators": 180,
            "models": ["linear_regression", "random_forest", "extra_trees"],
        },
        "accurate": {
            "rf_estimators": 320,
            "et_estimators": 420,
            "models": ["linear_regression", "random_forest", "extra_trees"],
        },
    }
    settings = profile_settings[training_profile]

    model_map = {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=settings["rf_estimators"],
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=settings["et_estimators"],
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return {
        model_name: model_map[model_name] for model_name in settings["models"]
    }


def sample_training_data(
    df: pd.DataFrame,
    max_training_rows: int,
    validation_strategy: str,
    date_column: str | None,
    random_state: int,
) -> pd.DataFrame:
    if max_training_rows <= 0 or len(df) <= max_training_rows:
        return df

    if validation_strategy == "time_series" and date_column and date_column in df.columns:
        ordered = df.copy()
        ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce")
        ordered = ordered.sort_values(date_column)
        return ordered.tail(max_training_rows).reset_index(drop=True)

    return df.sample(n=max_training_rows, random_state=random_state).reset_index(drop=True)


def build_time_series_features(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    group_columns: list[str],
    lag_steps: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    if not date_column:
        raise ValueError("Choose a date column for time-series validation.")

    prepared = df.copy()
    prepared[date_column] = pd.to_datetime(prepared[date_column], errors="coerce")
    prepared = prepared.dropna(subset=[date_column]).sort_values(
        group_columns + [date_column]
    )

    series_group = (
        prepared.groupby(group_columns, dropna=False)[target_column]
        if group_columns
        else prepared[target_column]
    )

    generated_columns: list[str] = []
    for lag in lag_steps:
        column_name = f"{target_column}_lag_{lag}"
        prepared[column_name] = (
            series_group.shift(lag)
            if group_columns
            else series_group.shift(lag)
        )
        generated_columns.append(column_name)

    for window in rolling_windows:
        mean_column = f"{target_column}_rolling_mean_{window}"
        std_column = f"{target_column}_rolling_std_{window}"
        if group_columns:
            prepared[mean_column] = series_group.transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )
            prepared[std_column] = series_group.transform(
                lambda values: values.shift(1).rolling(window, min_periods=2).std()
            )
        else:
            shifted_values = series_group.shift(1)
            prepared[mean_column] = shifted_values.rolling(window, min_periods=1).mean()
            prepared[std_column] = shifted_values.rolling(window, min_periods=2).std()
        generated_columns.extend([mean_column, std_column])

    if generated_columns:
        prepared = prepared.dropna(subset=generated_columns, how="all")

    return prepared.reset_index(drop=True)


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    validation_strategy: str,
    test_size: float,
    random_state: int,
    date_series: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    if validation_strategy == "time_series":
        if date_series is None:
            raise ValueError("Time-series validation requires a valid date column.")

        ordered_index = date_series.sort_values(kind="stable").index
        x_sorted = x.loc[ordered_index].reset_index(drop=True)
        y_sorted = y.loc[ordered_index].reset_index(drop=True)
        date_sorted = date_series.loc[ordered_index].reset_index(drop=True)

        split_index = max(1, int(len(x_sorted) * (1 - test_size)))
        split_index = min(split_index, len(x_sorted) - 1)
        if split_index <= 0 or split_index >= len(x_sorted):
            raise ValueError("Not enough rows for a time-series train/test split.")

        return (
            x_sorted.iloc[:split_index].copy(),
            x_sorted.iloc[split_index:].copy(),
            y_sorted.iloc[:split_index].copy(),
            y_sorted.iloc[split_index:].copy(),
            {
                "validation_strategy": "time_series",
                "train_start": str(date_sorted.iloc[0]),
                "train_end": str(date_sorted.iloc[split_index - 1]),
                "test_start": str(date_sorted.iloc[split_index]),
                "test_end": str(date_sorted.iloc[len(date_sorted) - 1]),
            },
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return (
        x_train,
        x_test,
        y_train,
        y_test,
        {"validation_strategy": "random_split"},
    )


def evaluate_model(
    name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    return {
        "name": name,
        "model": model,
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "predictions": predictions,
    }


def run_time_series_backtesting(
    models: dict[str, Pipeline],
    x: pd.DataFrame,
    y: pd.Series,
    split_count: int,
) -> list[dict[str, Any]]:
    if len(x) < split_count + 2:
        return []

    splitter = TimeSeriesSplit(n_splits=split_count)
    summaries: list[dict[str, Any]] = []

    for name, model in models.items():
        fold_metrics = []
        for train_index, test_index in splitter.split(x):
            fold_model = clone(model)
            x_train = x.iloc[train_index]
            x_test = x.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            fold_model.fit(x_train, y_train)
            predictions = fold_model.predict(x_test)
            fold_metrics.append(
                {
                    "rmse": mean_squared_error(y_test, predictions) ** 0.5,
                    "mae": mean_absolute_error(y_test, predictions),
                    "r2": r2_score(y_test, predictions),
                }
            )

        if not fold_metrics:
            continue

        fold_df = pd.DataFrame(fold_metrics)
        summaries.append(
            {
                "name": name,
                "cv_rmse_mean": round(fold_df["rmse"].mean(), 4),
                "cv_rmse_std": round(fold_df["rmse"].std(ddof=0), 4),
                "cv_mae_mean": round(fold_df["mae"].mean(), 4),
                "cv_r2_mean": round(fold_df["r2"].mean(), 4),
            }
        )

    return sorted(summaries, key=lambda item: item["cv_rmse_mean"])


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        cleaned_names = []
        for name in names:
            cleaned_names.append(name.split("__", 1)[1] if "__" in name else name)
        return cleaned_names
    except Exception:
        return []


def extract_feature_importance(trained_pipeline: Pipeline) -> pd.DataFrame | None:
    estimator = trained_pipeline.named_steps["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance_values = abs(estimator.coef_)
    else:
        return None

    if not feature_names or len(feature_names) != len(importance_values):
        feature_names = [f"feature_{index}" for index in range(len(importance_values))]

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_saved_bundle(path_str: str) -> dict[str, Any]:
    return joblib.load(path_str)


def summarize_feature_metadata(df: pd.DataFrame) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    datetime_columns = set(infer_datetime_columns(df))

    for column in df.columns:
        series = df[column]
        info: dict[str, Any] = {"dtype": str(series.dtype)}

        if column in datetime_columns:
            parsed = pd.to_datetime(series, errors="coerce").dropna()
            info["kind"] = "datetime"
            if not parsed.empty:
                info["min"] = str(parsed.min().date())
                info["max"] = str(parsed.max().date())
        elif pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            info["kind"] = "numeric"
            if not clean.empty:
                info["min"] = float(clean.min())
                info["max"] = float(clean.max())
                info["default"] = float(clean.median())
        else:
            values = series.dropna().astype(str)
            info["kind"] = "categorical"
            top_values = values.value_counts().head(20).index.tolist()
            info["options"] = top_values
            info["default"] = top_values[0] if top_values else ""

        metadata[column] = info

    return metadata


def guess_target_column(df: pd.DataFrame) -> str | None:
    preferred_names = [
        "sales",
        "sale_price",
        "price",
        "revenue",
        "target",
        "y",
        "demand",
        "units_sold",
        "qty",
        "quantity",
    ]
    lower_name_map = {column.lower(): column for column in df.columns}

    for preferred in preferred_names:
        if preferred in lower_name_map:
            candidate = lower_name_map[preferred]
            numeric_candidate = pd.to_numeric(df[candidate], errors="coerce")
            if numeric_candidate.notna().mean() >= 0.7:
                return candidate

    numeric_columns = []
    for column in df.columns:
        numeric_candidate = pd.to_numeric(df[column], errors="coerce")
        if numeric_candidate.notna().mean() >= 0.7:
            numeric_columns.append(column)

    return numeric_columns[-1] if numeric_columns else None


def train_bundle(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
    test_size: float,
    source_details: list[str],
    validation_strategy: str,
    date_column: str | None,
    group_columns: list[str],
    lag_steps: list[int],
    rolling_windows: list[int],
    backtest_splits: int,
    training_profile: str,
    max_training_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_working_df = df.copy()
    numeric_target = pd.to_numeric(raw_working_df[target_column], errors="coerce")
    valid_target_ratio = float(numeric_target.notna().mean()) if len(numeric_target) else 0.0
    if valid_target_ratio < 0.7:
        raise ValueError(
            f"Target column '{target_column}' is not sufficiently numeric for regression. "
            "Please choose a numeric target column such as sales, price, or revenue."
        )
    raw_working_df[target_column] = numeric_target

    raw_feature_df = raw_working_df.drop(columns=[target_column])
    raw_feature_columns = raw_feature_df.columns.tolist()
    raw_feature_dtypes = {
        column: str(dtype) for column, dtype in raw_feature_df.dtypes.items()
    }

    working_df = raw_working_df.copy()
    if validation_strategy == "time_series":
        if not date_column:
            raise ValueError("Choose a date column for time-series validation.")
        working_df = build_time_series_features(
            df=working_df,
            target_column=target_column,
            date_column=date_column,
            group_columns=group_columns,
            lag_steps=lag_steps,
            rolling_windows=rolling_windows,
        )

    working_df = sample_training_data(
        df=working_df,
        max_training_rows=max_training_rows,
        validation_strategy=validation_strategy,
        date_column=date_column,
        random_state=random_state,
    )

    date_series = (
        pd.to_datetime(working_df[date_column], errors="coerce")
        if date_column and date_column in working_df.columns
        else None
    )
    working_df = enrich_datetime_columns(working_df)
    valid_target_mask = working_df[target_column].notna()
    working_df = working_df.loc[valid_target_mask].reset_index(drop=True)
    if date_series is not None:
        date_series = date_series.loc[valid_target_mask].reset_index(drop=True)

    x = working_df.drop(columns=[target_column])
    y = working_df[target_column]

    x_train, x_test, y_train, y_test, split_summary = split_train_test(
        x,
        y,
        validation_strategy=validation_strategy,
        test_size=test_size,
        random_state=random_state,
        date_series=date_series,
    )

    preprocessor = build_preprocessor(x)
    models = build_models(preprocessor, random_state, training_profile)
    evaluations = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test)
        for name, model in models.items()
    ]
    best_result = min(evaluations, key=lambda item: item["rmse"])
    backtesting_summary = (
        run_time_series_backtesting(models, x_train, y_train, backtest_splits)
        if validation_strategy == "time_series" and training_profile != "fast"
        else []
    )
    feature_importance_df = (
        extract_feature_importance(best_result["model"])
        if training_profile != "fast"
        else None
    )

    sample_predictions = x_test.copy()
    sample_predictions[f"actual_{target_column}"] = y_test.values
    sample_predictions[f"predicted_{target_column}"] = best_result["predictions"]

    metrics = {
        "target_column": target_column,
        "row_count": len(working_df),
        "max_training_rows": max_training_rows,
        "feature_count": x.shape[1],
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "sources": source_details,
        "split_summary": split_summary,
        "time_series_settings": {
            "date_column": date_column,
            "group_columns": group_columns,
            "lag_steps": lag_steps,
            "rolling_windows": rolling_windows,
            "backtest_splits": backtest_splits,
        },
        "training_profile": training_profile,
        "best_model": best_result["name"],
        "best_model_metrics": {
            "rmse": round(best_result["rmse"], 4),
            "mae": round(best_result["mae"], 4),
            "r2": round(best_result["r2"], 4),
        },
        "all_models": [
            {
                "name": result["name"],
                "rmse": round(result["rmse"], 4),
                "mae": round(result["mae"], 4),
                "r2": round(result["r2"], 4),
            }
            for result in sorted(evaluations, key=lambda item: item["rmse"])
        ],
        "backtesting": backtesting_summary,
    }

    bundle = {
        "model": best_result["model"],
        "target_column": target_column,
        "metrics": metrics,
        "training_columns": x.columns.tolist(),
        "raw_feature_columns": raw_feature_columns,
        "raw_feature_dtypes": raw_feature_dtypes,
        "feature_metadata": summarize_feature_metadata(raw_feature_df),
        "feature_importance": (
            feature_importance_df.to_dict(orient="records")
            if feature_importance_df is not None
            else []
        ),
        "uses_time_series_features": validation_strategy == "time_series",
    }
    return bundle, sample_predictions


def save_training_outputs(bundle: dict[str, Any], sample_predictions: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    load_saved_bundle.clear()
    METRICS_PATH.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    sample_predictions.head(100).to_csv(PREDICTIONS_PATH, index=False)


def build_manual_input_frame(columns: list[str], dtypes: dict[str, str]) -> pd.DataFrame:
    values: dict[str, Any] = {}
    for column in columns:
        dtype_name = dtypes.get(column, "object")
        if any(token in dtype_name for token in ["int", "float", "double"]):
            values[column] = st.text_input(
                f"{column} ({dtype_name})",
                key=f"manual_{column}",
            )
        else:
            values[column] = st.text_input(f"{column}", key=f"manual_{column}")
    return pd.DataFrame([values])


def build_manual_timeline_frame(
    input_df: pd.DataFrame,
    feature_metadata: dict[str, Any],
    preferred_date_column: str | None,
) -> tuple[pd.DataFrame, str | None]:
    if input_df.empty:
        return input_df, None

    datetime_candidates = [
        column for column, meta in feature_metadata.items() if meta.get("kind") == "datetime"
    ]
    date_column = preferred_date_column if preferred_date_column in input_df.columns else None
    if date_column is None:
        for column in datetime_candidates:
            if column in input_df.columns:
                date_column = column
                break

    enable_timeline = st.checkbox(
        "Generate forecast graph over time from these values",
        value=bool(date_column),
    )
    if not enable_timeline:
        return input_df, None

    periods = st.slider(
        "Forecast periods for this input",
        min_value=2,
        max_value=60,
        value=12,
        step=1,
    )
    frequency = st.selectbox(
        "Forecast frequency",
        options=["D", "W", "MS", "M"],
        format_func=lambda value: {
            "D": "Daily",
            "W": "Weekly",
            "MS": "Monthly (start)",
            "M": "Monthly (end)",
        }[value],
        key="manual_timeline_frequency",
    )

    timeline_df = pd.concat([input_df] * periods, ignore_index=True)
    if date_column:
        default_start = pd.Timestamp.today().normalize().date()
        start_date = st.date_input(
            "Forecast start date",
            value=default_start,
            key="manual_timeline_start",
        )
        timeline_df[date_column] = pd.date_range(
            start=start_date,
            periods=periods,
            freq=frequency,
        ).astype(str)
        return timeline_df, date_column

    timeline_df["forecast_step"] = list(range(1, periods + 1))
    return timeline_df, "forecast_step"


def build_prediction_chart_data(
    results: pd.DataFrame,
    prediction_chart_column: str | None,
) -> tuple[pd.DataFrame, str]:
    chart_df = results.copy()

    if prediction_chart_column and prediction_chart_column in chart_df.columns:
        x_column = prediction_chart_column
        if prediction_chart_column != "forecast_step":
            chart_df[prediction_chart_column] = pd.to_datetime(
                chart_df[prediction_chart_column], errors="coerce"
            )
        chart_df = chart_df.sort_values(prediction_chart_column).reset_index(drop=True)
    else:
        x_column = "forecast_step"
        chart_df[x_column] = range(1, len(chart_df) + 1)

    if len(chart_df) > 1:
        window = min(5, len(chart_df))
        chart_df["prediction_trend"] = (
            chart_df["prediction"].rolling(window=window, min_periods=1).mean()
        )
    else:
        chart_df["prediction_trend"] = chart_df["prediction"]

    return chart_df, x_column


def render_prediction_visuals(
    results: pd.DataFrame,
    target_column: str,
    prediction_chart_column: str | None,
) -> None:
    if results.empty:
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows Predicted", f"{len(results)}")
    metric_cols[1].metric("Average", f"{float(results['prediction'].mean()):,.4f}")
    metric_cols[2].metric("Min", f"{float(results['prediction'].min()):,.4f}")
    metric_cols[3].metric("Max", f"{float(results['prediction'].max()):,.4f}")

    if len(results) == 1:
        st.metric(f"Predicted {target_column}", f"{float(results['prediction'].iloc[0]):,.4f}")
        return

    chart_df, x_column = build_prediction_chart_data(results, prediction_chart_column)
    base = alt.Chart(chart_df).encode(
        x=alt.X(x_column, title="Time" if x_column != "forecast_step" else "Step"),
        tooltip=[
            alt.Tooltip(x_column, title="Time" if x_column != "forecast_step" else "Step"),
            alt.Tooltip("prediction:Q", format=",.4f", title="Prediction"),
            alt.Tooltip("prediction_trend:Q", format=",.4f", title="Trend"),
        ],
    )

    line = base.mark_line(color="#0f766e", strokeWidth=3).encode(
        y=alt.Y("prediction:Q", title=f"Predicted {target_column}")
    )
    points = base.mark_circle(color="#0f766e", size=70).encode(y="prediction:Q")
    trend = base.mark_line(color="#f97316", strokeDash=[6, 4], strokeWidth=2).encode(
        y=alt.Y("prediction_trend:Q", title=f"Predicted {target_column}")
    )

    st.subheader("Prediction Graph")

    view_mode = st.radio(
        "Graph view",
        options=["line", "bar", "cumulative"],
        format_func=lambda value: {
            "line": "Line + trend",
            "bar": "Bar view",
            "cumulative": "Cumulative forecast",
        }[value],
        horizontal=True,
        key="prediction_graph_mode",
    )

    if view_mode == "line":
        st.altair_chart((line + points + trend).interactive(), use_container_width=True)
    elif view_mode == "bar":
        bar_chart = alt.Chart(chart_df).mark_bar(color="#2563eb").encode(
            x=alt.X(x_column, title="Time" if x_column != "forecast_step" else "Step"),
            y=alt.Y("prediction:Q", title=f"Predicted {target_column}"),
            tooltip=[
                alt.Tooltip(x_column, title="Time" if x_column != "forecast_step" else "Step"),
                alt.Tooltip("prediction:Q", format=",.4f", title="Prediction"),
            ],
        )
        st.altair_chart(bar_chart.interactive(), use_container_width=True)
    elif view_mode == "cumulative":
        cumulative_df = chart_df.copy()
        cumulative_df["cumulative_prediction"] = cumulative_df["prediction"].cumsum()
        cumulative_chart = alt.Chart(cumulative_df).mark_area(
            color="#38bdf8", opacity=0.55
        ).encode(
            x=alt.X(x_column, title="Time" if x_column != "forecast_step" else "Step"),
            y=alt.Y("cumulative_prediction:Q", title="Cumulative Prediction"),
            tooltip=[
                alt.Tooltip(x_column, title="Time" if x_column != "forecast_step" else "Step"),
                alt.Tooltip("cumulative_prediction:Q", format=",.4f", title="Cumulative"),
            ],
        )
        st.altair_chart(cumulative_chart.interactive(), use_container_width=True)


def build_scenario_input_frame(
    raw_feature_columns: list[str],
    feature_metadata: dict[str, Any],
    time_series_settings: dict[str, Any],
) -> tuple[pd.DataFrame, str | None]:
    scenario_key = st.session_state.setdefault("scenario_form_key", str(uuid4()))
    date_column = time_series_settings.get("date_column")
    input_values: dict[str, Any] = {}

    st.caption("Set base values once, then generate a prediction timeline.")
    horizon = st.slider("Forecast periods", min_value=2, max_value=60, value=12, step=1)
    frequency = st.selectbox(
        "Time frequency",
        options=["D", "W", "MS", "M"],
        format_func=lambda value: {
            "D": "Daily",
            "W": "Weekly",
            "MS": "Monthly (start)",
            "M": "Monthly (end)",
        }[value],
    )

    start_date = None
    if date_column and date_column in raw_feature_columns:
        date_meta = feature_metadata.get(date_column, {})
        default_start = pd.Timestamp.today().normalize().date()
        if date_meta.get("max"):
            try:
                default_start = (
                    pd.to_datetime(date_meta["max"]).normalize() + pd.Timedelta(days=1)
                ).date()
            except Exception:
                pass
        start_date = st.date_input("Start date", value=default_start, key=f"start_{scenario_key}")

    for column in raw_feature_columns:
        if column == date_column:
            continue

        meta = feature_metadata.get(column, {})
        kind = meta.get("kind", "categorical")
        widget_key = f"{scenario_key}_{column}"

        if kind == "numeric":
            default_value = float(meta.get("default", 0.0))
            min_value = float(meta.get("min", default_value - 100)) if "min" in meta else None
            max_value = float(meta.get("max", default_value + 100)) if "max" in meta else None
            if min_value is not None and max_value is not None and min_value < max_value:
                input_values[column] = st.number_input(
                    column,
                    value=default_value,
                    min_value=min_value,
                    max_value=max_value,
                    key=widget_key,
                )
            else:
                input_values[column] = st.number_input(
                    column,
                    value=default_value,
                    key=widget_key,
                )
        elif kind == "datetime":
            default_date = pd.Timestamp.today().normalize().date()
            input_values[column] = str(
                st.date_input(column, value=default_date, key=widget_key)
            )
        else:
            options = meta.get("options", [])
            if options:
                input_values[column] = st.selectbox(column, options=options, key=widget_key)
            else:
                input_values[column] = st.text_input(column, key=widget_key)

    if date_column and start_date is not None:
        date_range = pd.date_range(start=start_date, periods=horizon, freq=frequency)
        scenario_df = pd.DataFrame({date_column: date_range.astype(str)})
    else:
        scenario_df = pd.DataFrame({"period_index": list(range(1, horizon + 1))})

    for column, value in input_values.items():
        scenario_df[column] = value

    if "period_index" in scenario_df.columns and "period_index" not in raw_feature_columns:
        scenario_df = scenario_df.drop(columns=["period_index"])

    return scenario_df, date_column


def coerce_prediction_input(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    coerced = df.copy()
    for column, dtype_name in dtypes.items():
        if column not in coerced.columns:
            continue
        if any(token in dtype_name for token in ["int", "float", "double"]):
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def align_prediction_frame(
    df: pd.DataFrame,
    raw_feature_dtypes: dict[str, str],
    training_columns: list[str],
) -> pd.DataFrame:
    aligned = coerce_prediction_input(df.copy(), raw_feature_dtypes)
    aligned = enrich_datetime_columns(aligned)
    for column in training_columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[training_columns]


def initialize_source_text() -> None:
    if "source_text" not in st.session_state:
        st.session_state["source_text"] = "\n".join(
            entry["url"] for entry in PROVIDED_KAGGLE_SOURCES
        )


def dataset_fetcher_section() -> None:
    st.subheader("1. Fetch Kaggle Data")
    initialize_source_text()

    st.caption("Built-in Kaggle sources from your earlier list")
    selected_preset_labels = st.multiselect(
        "Choose built-in Kaggle sources",
        options=[entry["label"] for entry in PROVIDED_KAGGLE_SOURCES],
        default=[entry["label"] for entry in PROVIDED_KAGGLE_SOURCES],
    )

    selected_preset_sources = [
        entry for entry in PROVIDED_KAGGLE_SOURCES if entry["label"] in selected_preset_labels
    ]
    if selected_preset_sources:
        for entry in selected_preset_sources:
            st.markdown(
                f"- `{entry['label']}` [{entry['kind']}]  "
                f"[{entry['url']}]({entry['url']})"
            )

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("Load Selected Presets"):
        st.session_state["source_text"] = "\n".join(
            entry["url"] for entry in selected_preset_sources
        )
    if action_col2.button("Load All Provided Sources"):
        st.session_state["source_text"] = "\n".join(
            entry["url"] for entry in PROVIDED_KAGGLE_SOURCES
        )

    source_text = st.text_area(
        "Paste one Kaggle dataset link per line",
        key="source_text",
        placeholder=(
            "https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast\n"
            "https://www.kaggle.com/datasets/tevecsystems/retail-sales-forecasting"
        ),
        height=140,
    )
    sources = [line.strip() for line in source_text.splitlines() if line.strip()]

    test_col1, test_col2 = st.columns([1, 2])
    if test_col1.button("Test Kaggle Connection"):
        try:
            with st.spinner("Testing Kaggle authentication..."):
                connection_summary = test_kaggle_connection()
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success("Kaggle authentication is working.")
            with test_col2:
                st.json(connection_summary)

    fetch_workers = st.slider(
        "Parallel fetch workers",
        min_value=1,
        max_value=4,
        value=3,
        step=1,
        help="Higher values can speed up multiple source downloads when your connection allows it.",
    )
    preview_rows = st.slider(
        "Preview rows to load for CSV inspection",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Lower values make file previews render faster.",
    )
    st.session_state["preview_rows"] = int(preview_rows)

    cache_statuses = [source_cache_status(source) for source in sources]
    if cache_statuses:
        with st.expander("Source Cache Status", expanded=False):
            status_df = pd.DataFrame(cache_statuses)[
                ["identifier", "source_kind", "cached", "csv_count"]
            ]
            st.dataframe(status_df, use_container_width=True)

    fetch_col1, fetch_col2 = st.columns(2)
    fetch_all_clicked = fetch_col1.button("Fetch Kaggle Sources", type="primary")
    fetch_missing_clicked = fetch_col2.button("Fetch Only Missing Sources")

    if fetch_all_clicked or fetch_missing_clicked:
        if not sources:
            st.warning("Add at least one Kaggle source first.")
            return

        sources_to_fetch = sources
        if fetch_missing_clicked:
            sources_to_fetch = [
                status["source"] for status in cache_statuses if not status["cached"]
            ]
            if not sources_to_fetch:
                st.success("All selected sources are already cached.")
                return

        progress_bar = st.progress(0)
        progress_status = st.empty()
        source_status_box = st.empty()
        per_source_updates = [
            f"- {source_cache_status(source)['identifier']}: queued" for source in sources_to_fetch
        ]
        source_status_box.markdown("\n".join(per_source_updates))

        try:
            fetched_by_source: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=min(int(fetch_workers), len(sources_to_fetch))) as executor:
                future_to_source = {
                    executor.submit(fetch_kaggle_source_uncached, source): source
                    for source in sources_to_fetch
                }
                total_sources = len(future_to_source)
                completed_sources = 0
                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    result = future.result()
                    fetched_by_source[source] = result
                    completed_sources += 1
                    result_label = result["identifier"]
                    result_state = "cache" if result.get("from_cache") else "downloaded"
                    progress_status.info(
                        f"Fetched {completed_sources}/{total_sources}: {result_label} [{result_state}]"
                    )
                    per_source_updates = [
                        (
                            f"- {source_cache_status(item)['identifier']}: "
                            f"{('cache' if fetched_by_source[item].get('from_cache') else 'downloaded')}"
                            if item in fetched_by_source
                            else f"- {source_cache_status(item)['identifier']}: queued"
                        )
                        for item in sources_to_fetch
                    ]
                    source_status_box.markdown("\n".join(per_source_updates))
                    progress_bar.progress(completed_sources / total_sources)
        except Exception as exc:
            st.error(str(exc))
            return

        if fetch_missing_clicked:
            cached_existing = [
                fetch_kaggle_source_uncached(status["source"])
                for status in cache_statuses
                if status["cached"]
            ]
            fetched_sources = []
            for source in sources:
                if source in fetched_by_source:
                    fetched_sources.append(fetched_by_source[source])
                else:
                    fetched_sources.append(
                        next(item for item in cached_existing if item["source"] == source)
                    )
        else:
            fetched_sources = [fetched_by_source[source] for source in sources]

        st.session_state["fetched_sources"] = fetched_sources
        cached_count = sum(1 for entry in fetched_sources if entry.get("from_cache"))
        downloaded_count = len(fetched_sources) - cached_count
        st.success(
            f"Fetched {len(fetched_sources)} Kaggle source(s). "
            f"Cached: {cached_count}, downloaded: {downloaded_count}."
        )

    fetched_sources = st.session_state.get("fetched_sources", [])
    if fetched_sources:
        st.caption("Fetched sources")
        for entry in fetched_sources:
            st.markdown(
                f"- `{entry['identifier']}` ({entry['source_kind']}) with "
                f"`{len(entry['csv_files'])}` CSV file(s) "
                f"[{'cache' if entry.get('from_cache') else 'download'}]"
            )


def training_section() -> None:
    fetched_sources = st.session_state.get("fetched_sources", [])
    if not fetched_sources:
        st.info("Fetch at least one Kaggle source to continue.")
        return

    st.subheader("2. Combine Files And Train")

    csv_lookup: dict[str, str] = {}
    csv_options: list[str] = []
    for source_entry in fetched_sources:
        base_dir = Path(source_entry["directory"])
        for csv_path_str in source_entry["csv_files"]:
            csv_path = Path(csv_path_str)
            label = f"{source_entry['identifier']} :: {csv_path.relative_to(base_dir)}"
            csv_lookup[label] = csv_path_str
            csv_options.append(label)

    selected_labels = st.multiselect(
        "Choose CSV files to combine for training",
        options=csv_options,
        default=csv_options[:1],
    )
    merge_strategy = st.selectbox(
        "Combine strategy",
        options=["common_columns", "keep_all_columns"],
        format_func=lambda value: {
            "common_columns": "Keep only columns shared by every selected file",
            "keep_all_columns": "Keep all columns and fill missing values automatically",
        }[value],
    )

    preview_target = None
    preview_df = None
    common_columns: list[str] = []
    if selected_labels:
        selected_paths = tuple(csv_lookup[label] for label in selected_labels)
        common_columns = get_common_columns(selected_paths)

        if not common_columns:
            st.error("The selected CSV files do not share any common columns.")
            return

        preview_df = load_csv_preview(
            csv_lookup[selected_labels[0]],
            nrows=int(st.session_state.get("preview_rows", 20)),
        )
        preview_metadata = get_csv_metadata(csv_lookup[selected_labels[0]])
        preview_common_df = preview_df[[column for column in common_columns if column in preview_df.columns]]
        guessed_target = guess_target_column(preview_common_df)
        default_target_index = (
            common_columns.index(guessed_target)
            if guessed_target in common_columns
            else 0
        )
        preview_target = st.selectbox(
            "Target column to predict",
            options=common_columns,
            index=default_target_index,
            help="Only columns shared by all selected CSV files are shown here.",
        )
        st.caption(
            f"Previewing {preview_metadata['preview_rows']} rows from the first selected CSV for faster inspection."
        )
        st.caption(f"Common columns across selected files: `{len(common_columns)}`")
        if guessed_target:
            st.caption(f"Suggested target column: `{guessed_target}`")
        st.dataframe(preview_df.head(5), use_container_width=True)

    validation_strategy = st.selectbox(
        "Validation strategy",
        options=["random_split", "time_series"],
        format_func=lambda value: {
            "random_split": "Random train/test split",
            "time_series": "Time-based split with backtesting",
        }[value],
    )

    date_column = None
    group_columns: list[str] = []
    lag_steps: list[int] = []
    rolling_windows: list[int] = []
    backtest_splits = 3
    if validation_strategy == "time_series" and preview_df is not None:
        candidate_date_columns = [
            column for column in infer_datetime_columns(preview_df) if column in common_columns
        ]
        date_options = candidate_date_columns or common_columns
        default_date_index = 0 if date_options else None
        date_column = st.selectbox(
            "Date column",
            options=date_options,
            index=default_date_index,
        )
        group_columns = st.multiselect(
            "Group columns for lag features",
            options=[
                column
                for column in common_columns
                if column not in {preview_target, date_column}
            ],
            help="Use these when the dataset contains multiple stores, items, or categories.",
        )
        lag_steps = st.multiselect(
            "Lag steps",
            options=[1, 2, 3, 7, 14, 28],
            default=[1, 7, 14],
        )
        rolling_windows = st.multiselect(
            "Rolling windows",
            options=[3, 7, 14, 28],
            default=[7, 14],
        )
        backtest_splits = st.slider(
            "Backtesting splits",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
        )

    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    training_profile = st.selectbox(
        "Training speed profile",
        options=["fast", "balanced", "accurate"],
        index=0,
        format_func=lambda value: {
            "fast": "Fast load and train",
            "balanced": "Balanced",
            "accurate": "Higher accuracy, slower",
        }[value],
    )
    max_training_rows = st.number_input(
        "Max training rows",
        min_value=0,
        value=20000,
        step=1000,
        help="Set 0 to use all rows. Lower values train much faster on large datasets.",
    )

    if st.button("Train Model"):
        if not selected_labels:
            st.warning("Select at least one CSV file.")
            return
        if not preview_target:
            st.warning("Choose a target column.")
            return

        try:
            frames = [load_csv_file(csv_lookup[label]) for label in selected_labels]
            combined_df, columns_used = combine_dataframes(
                frames=frames,
                source_labels=selected_labels,
                target_column=preview_target,
                merge_strategy=merge_strategy,
            )
            bundle, sample_predictions = train_bundle(
                df=combined_df,
                target_column=preview_target,
                random_state=int(random_state),
                test_size=float(test_size),
                source_details=selected_labels,
                validation_strategy=validation_strategy,
                date_column=date_column,
                group_columns=group_columns,
                lag_steps=lag_steps,
                rolling_windows=rolling_windows,
                backtest_splits=backtest_splits,
                training_profile=training_profile,
                max_training_rows=int(max_training_rows),
            )
            save_training_outputs(bundle, sample_predictions)
        except Exception as exc:
            st.error(str(exc))
            return

        st.session_state["trained_bundle"] = bundle
        st.session_state["columns_used"] = columns_used

        st.success(
            f"Training completed. Best model: {bundle['metrics']['best_model']} "
            f"(RMSE {bundle['metrics']['best_model_metrics']['rmse']})"
        )
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame(bundle["metrics"]["all_models"])
        st.dataframe(comparison_df, use_container_width=True)

        split_summary = bundle["metrics"].get("split_summary", {})
        if split_summary:
            st.subheader("Split Summary")
            st.json(split_summary)

        backtesting = bundle["metrics"].get("backtesting", [])
        if backtesting:
            st.subheader("Backtesting")
            st.dataframe(pd.DataFrame(backtesting), use_container_width=True)

        feature_importance = bundle.get("feature_importance", [])
        if feature_importance:
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame(feature_importance).head(20).set_index("feature")
            st.bar_chart(importance_df)

        with st.expander("Full Metrics JSON"):
            st.json(bundle["metrics"])
        st.dataframe(sample_predictions.head(20), use_container_width=True)

        metrics_bytes = json.dumps(bundle["metrics"], indent=2).encode("utf-8")
        st.download_button(
            "Download Metrics JSON",
            data=metrics_bytes,
            file_name="metrics.json",
            mime="application/json",
        )
        csv_bytes = sample_predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Sample Predictions CSV",
            data=csv_bytes,
            file_name="sample_predictions.csv",
            mime="text/csv",
        )


def prediction_section() -> None:
    st.subheader("3. Predict")

    bundle = st.session_state.get("trained_bundle")
    if bundle is None and MODEL_BUNDLE_PATH.exists():
        bundle = load_saved_bundle(str(MODEL_BUNDLE_PATH))
        st.session_state["trained_bundle"] = bundle

    if bundle is None:
        st.info("Train a model first or keep the saved model bundle in the models folder.")
        return

    model = bundle["model"]
    training_columns = bundle["training_columns"]
    raw_feature_columns = bundle.get("raw_feature_columns", training_columns)
    raw_feature_dtypes = bundle.get("raw_feature_dtypes", {})
    feature_metadata = bundle.get("feature_metadata", {})
    time_series_settings = bundle["metrics"].get("time_series_settings", {})
    st.caption(f"Loaded target: {bundle['target_column']}")
    if bundle.get("uses_time_series_features"):
        st.warning(
            "This model was trained with time-series lag features. Future prediction works best when your uploaded CSV includes enough historical context or precomputed lag columns."
        )

    prediction_mode = st.radio(
        "Prediction input mode",
        options=["upload_csv", "manual_form", "scenario_over_time"],
        format_func=lambda value: {
            "upload_csv": "Upload a CSV for batch predictions",
            "manual_form": "Enter one record manually",
            "scenario_over_time": "Set values and plot predictions over time",
        }[value],
        horizontal=True,
    )

    input_df = None
    prediction_chart_column = None
    if prediction_mode == "upload_csv":
        uploaded_file = st.file_uploader("Upload prediction CSV", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            candidate_chart_columns = [
                column
                for column, meta in feature_metadata.items()
                if meta.get("kind") == "datetime" and column in input_df.columns
            ]
            if candidate_chart_columns:
                prediction_chart_column = candidate_chart_columns[0]
    elif prediction_mode == "scenario_over_time":
        input_df, prediction_chart_column = build_scenario_input_frame(
            raw_feature_columns=raw_feature_columns,
            feature_metadata=feature_metadata,
            time_series_settings=time_series_settings,
        )
        st.dataframe(input_df.head(20), use_container_width=True)
    else:
        input_df = build_manual_input_frame(raw_feature_columns, raw_feature_dtypes)
        input_df, prediction_chart_column = build_manual_timeline_frame(
            input_df=input_df,
            feature_metadata=feature_metadata,
            preferred_date_column=time_series_settings.get("date_column"),
        )

    if input_df is not None and st.button("Run Prediction"):
        try:
            aligned_df = align_prediction_frame(
                input_df,
                raw_feature_dtypes=raw_feature_dtypes,
                training_columns=training_columns,
            )
            predictions = model.predict(aligned_df)
            results = input_df.copy()
            results["prediction"] = predictions
        except Exception as exc:
            st.error(str(exc))
            return

        st.session_state["prediction_results"] = results
        st.session_state["prediction_target_column"] = bundle["target_column"]
        st.session_state["prediction_chart_column"] = prediction_chart_column

    # Render from session_state so the graph persists across reruns
    # (e.g. when the user switches the graph view radio button)
    if "prediction_results" in st.session_state:
        results = st.session_state["prediction_results"]
        stored_target = st.session_state.get("prediction_target_column", bundle["target_column"])
        stored_chart_col = st.session_state.get("prediction_chart_column")

        render_prediction_visuals(
            results=results,
            target_column=stored_target,
            prediction_chart_column=stored_chart_col,
        )
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download Predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


def model_notes_section() -> None:
    st.subheader("Notes")
    st.markdown(
        "- The app now supports both random validation and time-series validation.\n"
        "- Time-series mode can create lag and rolling-window features from the target column.\n"
        "- Backtesting metrics are shown for time-series mode to compare models more realistically.\n"
        "- The prediction area now supports scenario-based value entry and prediction charts over time.\n"
        "- If you combine files from different datasets, accuracy still depends on matching target meaning and compatible columns."
    )


def main() -> None:
    st.set_page_config(page_title="Sales Forecast Trainer", layout="wide")
    st.title("Sales Forecast Training App")
    st.write(
        "Fetch Kaggle data, combine CSV files, train a regression model, and generate predictions from one Streamlit app."
    )

    dataset_fetcher_section()
    training_section()
    prediction_section()
    model_notes_section()


if __name__ == "__main__":
    main()
