from __future__ import annotations

from datetime import date
from pathlib import Path
import json

import pandas as pd


def load_refined_frame_with_scaler(
    *,
    processed_root_dir: Path,
    source: str,
    symbol: str,
    extraction_date: date,
    lookback: int,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    manifest_path = (
        processed_root_dir
        / "manifests"
        / f"extraction_date={extraction_date.isoformat()}"
        / "refined_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Refined manifest not found: {manifest_path}. "
            "Run `python scripts/generate_refined.py --skip-s3` first."
        )

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_request = manifest_payload.get("request", {})
    if manifest_request.get("source") != source:
        raise ValueError(
            f"Refined manifest source {manifest_request.get('source')!r} "
            f"does not match requested source {source!r}."
        )
    if manifest_request.get("target_column") != target_column:
        raise ValueError(
            f"Refined manifest target_column {manifest_request.get('target_column')!r} "
            f"does not match requested target_column {target_column!r}."
        )

    asset_payload = next(
        (
            asset
            for asset in manifest_payload.get("assets", [])
            if asset.get("symbol", "").upper() == symbol.upper()
            and int(asset.get("feature_count", lookback)) == lookback
        ),
        None,
    )
    if asset_payload is None:
        raise ValueError(
            f"Could not find refined asset metadata for symbol {symbol!r} "
            f"with lookback={lookback} in {manifest_path}."
        )

    refined_path = Path(str(asset_payload.get("local_path", "")))
    if not refined_path.exists():
        refined_path = (
            processed_root_dir
            / "refined_data"
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / "refined.parquet"
        )
    if not refined_path.exists():
        raise FileNotFoundError(
            f"Refined dataset not found for symbol {symbol!r}: {refined_path}"
        )

    frame = pd.read_parquet(refined_path)
    scaler_metadata = {
        "min_offset": float(asset_payload["scaler_min_offset"]),
        "scale": float(asset_payload["scaler_scale"]),
        "data_min": float(asset_payload.get("data_min", 0.0)),
        "data_max": float(asset_payload.get("data_max", 0.0)),
    }
    return frame, scaler_metadata


def load_feature_frame_with_scaler(
    *,
    processed_root_dir: Path,
    source: str,
    symbol: str,
    extraction_date: date,
    lookback: int,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, float]] | None:
    manifest_path = (
        processed_root_dir
        / "manifests"
        / f"extraction_date={extraction_date.isoformat()}"
        / "feature_manifest.json"
    )
    if not manifest_path.exists():
        return None

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_request = manifest_payload.get("request", {})
    if (
        manifest_request.get("source") != source
        or manifest_request.get("target_column") != target_column
    ):
        return None

    asset_payload = next(
        (
            asset
            for asset in manifest_payload.get("assets", [])
            if asset.get("symbol", "").upper() == symbol.upper()
            and int(asset.get("lookback", lookback)) == lookback
        ),
        None,
    )
    if asset_payload is None:
        return None

    feature_path = Path(str(asset_payload.get("local_path", "")))
    if not feature_path.exists():
        feature_path = (
            processed_root_dir
            / "feature_data"
            / f"source={source}"
            / f"symbol={symbol.upper()}"
            / f"lookback={lookback}"
            / f"extraction_date={extraction_date.isoformat()}"
            / "features.parquet"
        )
    if not feature_path.exists():
        return None

    frame = pd.read_parquet(feature_path)
    scaler_metadata = {
        "min_offset": float(asset_payload["scaler_min_offset"]),
        "scale": float(asset_payload["scaler_scale"]),
        "data_min": float(asset_payload.get("data_min", 0.0)),
        "data_max": float(asset_payload.get("data_max", 0.0)),
    }
    return frame, scaler_metadata


def load_preferred_training_frame(
    *,
    processed_root_dir: Path,
    source: str,
    symbol: str,
    extraction_date: date,
    lookback: int,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, float], str]:
    feature_payload = load_feature_frame_with_scaler(
        processed_root_dir=processed_root_dir,
        source=source,
        symbol=symbol,
        extraction_date=extraction_date,
        lookback=lookback,
        target_column=target_column,
    )
    if feature_payload is not None:
        frame, scaler_metadata = feature_payload
        return frame, scaler_metadata, "feature"

    frame, scaler_metadata = load_refined_frame_with_scaler(
        processed_root_dir=processed_root_dir,
        source=source,
        symbol=symbol,
        extraction_date=extraction_date,
        lookback=lookback,
        target_column=target_column,
    )
    return frame, scaler_metadata, "refined"
