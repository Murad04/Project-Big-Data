from __future__ import annotations

import argparse
from pathlib import Path

from ..ml.train import train_catboost_model
from .preprocess import download_public_datasets, load_and_prepare_training_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CatBoost model for flight delay prediction.")
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory used for downloaded raw datasets.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory used to save trained model files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory used to save training metrics and reports.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=150000,
        help="Optional limit for training rows to speed up experimentation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    artifacts_dir = Path(args.artifacts_dir)

    flights_path, weather_path = download_public_datasets(data_dir=data_dir)
    df = load_and_prepare_training_data(
        flights_path=flights_path,
        weather_path=weather_path,
        max_rows=args.max_rows,
    )

    result = train_catboost_model(
        df=df,
        models_dir=models_dir,
        artifacts_dir=artifacts_dir,
    )

    print("Training complete")
    print(f"model: {result.model_name}")
    print(f"model_path: {result.model_path}")
    print(f"metrics_path: {result.metrics_path}")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
