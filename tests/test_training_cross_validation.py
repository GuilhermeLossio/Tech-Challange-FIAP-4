from __future__ import annotations

import pytest

from src.application.use_cases.train_model import KerasTrainingService


def test_time_series_cv_slices_are_expanding_and_hold_out_future_rows() -> None:
    slices = KerasTrainingService._build_time_series_cv_slices(
        sample_count=100,
        requested_folds=3,
        min_train_size=40,
    )

    assert slices == [
        (1, slice(0, 40), slice(40, 60)),
        (2, slice(0, 60), slice(60, 80)),
        (3, slice(0, 80), slice(80, 100)),
    ]


def test_time_series_cv_rejects_single_fold() -> None:
    with pytest.raises(ValueError, match="0 or at least 2"):
        KerasTrainingService._build_time_series_cv_slices(
            sample_count=100,
            requested_folds=1,
            min_train_size=40,
        )


def test_time_series_cv_rejects_empty_validation_windows() -> None:
    with pytest.raises(ValueError, match="too few rows"):
        KerasTrainingService._build_time_series_cv_slices(
            sample_count=10,
            requested_folds=3,
            min_train_size=9,
        )
