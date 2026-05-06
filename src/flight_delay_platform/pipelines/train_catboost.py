from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..ml.train import train_catboost_model
from .preprocess import download_public_datasets, load_and_prepare_training_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CatBoost model for flight delay prediction.")
    parser.add_argument("--data-dir",      default="data/raw",       help="Directory for raw CSV downloads.")
    parser.add_argument("--models-dir",    default="models",         help="Directory to save model files.")
    parser.add_argument("--artifacts-dir", default="artifacts",      help="Directory to save metrics/reports.")
    parser.add_argument("--max-rows",      type=int, default=0,      help="Row limit for preprocessing (0 = all rows).")
    parser.add_argument(
        "--processed-data",
        default=None,
        metavar="PATH",
        help="Path to a pre-processed Parquet directory (output of the Dask pipeline). "
             "Skips all CSV preprocessing when provided.",
    )
    parser.add_argument(
        "--use-dask",
        action="store_true",
        help="Run the Dask preprocessing pipeline first, then train from its Parquet output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    models_dir    = Path(args.models_dir)
    artifacts_dir = Path(args.artifacts_dir)

    # ── Decide data source ────────────────────────────────────────────────────
    if args.use_dask:
        from .dask_preprocess import run_dask_pipeline
        from .preprocess import download_public_datasets

        data_dir   = Path(args.data_dir)
        output_dir = Path("data/processed")

        print("=" * 60)
        print("STEP 1 - Dask Feature Engineering Pipeline")
        print("=" * 60)
        flights_path, weather_path = download_public_datasets(data_dir=data_dir)
        run_dask_pipeline(
            flights_path=flights_path,
            weather_path=weather_path,
            output_path=output_dir,
            max_rows=args.max_rows,
        )
        parquet_path = output_dir / "features.parquet"
        print("\n" + "=" * 60)
        print("STEP 2 - CatBoost Model Training")
        print("=" * 60)
        df = pd.read_parquet(str(parquet_path))
        print(f"Loaded {len(df):,} rows from Dask Parquet output")

    elif args.processed_data:
        parquet_path = Path(args.processed_data)
        print(f"Loading pre-processed Parquet from {parquet_path}…")
        df = pd.read_parquet(str(parquet_path))
        print(f"Loaded {len(df):,} rows")

    else:
        data_dir = Path(args.data_dir)
        flights_path, weather_path = download_public_datasets(data_dir=data_dir)
        df = load_and_prepare_training_data(
            flights_path=flights_path,
            weather_path=weather_path,
            max_rows=args.max_rows,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    result = train_catboost_model(df=df, models_dir=models_dir, artifacts_dir=artifacts_dir)

    print("\nTraining complete")
    print(f"model      : {result.model_name}")
    print(f"model_path : {result.model_path}")
    for key, value in result.metrics.items():
        print(f"{key:30s}: {value}")


if __name__ == "__main__":
    main()