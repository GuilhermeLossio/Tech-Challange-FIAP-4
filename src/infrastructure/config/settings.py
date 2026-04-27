from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _strip_matching_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(env_path: Path = DEFAULT_ENV_PATH) -> bool:
    """Load a simple .env file without introducing extra dependencies."""
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_matching_quotes(value)
        if key:
            os.environ.setdefault(key, value)

    return True


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _normalize_s3_prefix(prefix: str | None) -> str:
    if prefix is None:
        return "raw"
    normalized = prefix.strip().strip("/")
    return normalized or "raw"


@dataclass(frozen=True)
class RawPipelineSettings:
    local_raw_dir: Path
    s3_bucket_raw: str | None
    s3_raw_prefix: str
    aws_region: str | None
    aws_endpoint_url: str | None

    @classmethod
    def from_env(cls) -> "RawPipelineSettings":
        load_env_file()

        return cls(
            local_raw_dir=_resolve_project_path(
                os.getenv("RAW_LOCAL_DIR", "data/raw")
            ),
            s3_bucket_raw=os.getenv("S3_BUCKET_RAW"),
            s3_raw_prefix=_normalize_s3_prefix(os.getenv("S3_RAW_PREFIX", "raw")),
            aws_region=os.getenv("AWS_REGION"),
            aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )

    @property
    def s3_enabled(self) -> bool:
        return bool(self.s3_bucket_raw)


@dataclass(frozen=True)
class RefinedPipelineSettings:
    local_raw_dir: Path
    local_processed_dir: Path
    s3_bucket_refined: str | None
    s3_refined_prefix: str
    aws_region: str | None
    aws_endpoint_url: str | None

    @classmethod
    def from_env(cls) -> "RefinedPipelineSettings":
        load_env_file()

        return cls(
            local_raw_dir=_resolve_project_path(
                os.getenv("RAW_LOCAL_DIR", "data/raw")
            ),
            local_processed_dir=_resolve_project_path(
                os.getenv("PROCESSED_LOCAL_DIR", "data/processed")
            ),
            s3_bucket_refined=(
                os.getenv("S3_BUCKET_REFINED") or os.getenv("S3_BUCKET_RAW")
            ),
            s3_refined_prefix=_normalize_s3_prefix(
                os.getenv("S3_REFINED_PREFIX", "refined")
            ),
            aws_region=os.getenv("AWS_REGION"),
            aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )

    @property
    def s3_enabled(self) -> bool:
        return bool(self.s3_bucket_refined)


@dataclass(frozen=True)
class TrainingPipelineSettings:
    local_processed_dir: Path
    local_models_dir: Path
    s3_bucket_processed: str | None
    s3_processed_prefix: str
    s3_bucket_model: str | None
    s3_model_prefix: str
    aws_region: str | None
    aws_endpoint_url: str | None

    @classmethod
    def from_env(cls) -> "TrainingPipelineSettings":
        load_env_file()

        return cls(
            local_processed_dir=_resolve_project_path(
                os.getenv("PROCESSED_LOCAL_DIR", "data/processed")
            ),
            local_models_dir=_resolve_project_path(
                os.getenv("MODELS_DIR", "models")
            ),
            s3_bucket_processed=(
                os.getenv("S3_BUCKET_PROCESSED")
                or os.getenv("S3_BUCKET_REFINED")
                or os.getenv("S3_BUCKET_RAW")
            ),
            s3_processed_prefix=_normalize_s3_prefix(
                os.getenv("S3_PROCESSED_PREFIX", "processed")
            ),
            s3_bucket_model=(
                os.getenv("S3_BUCKET_MODEL")
                or os.getenv("S3_BUCKET_PROCESSED")
                or os.getenv("S3_BUCKET_REFINED")
                or os.getenv("S3_BUCKET_RAW")
            ),
            s3_model_prefix=_normalize_s3_prefix(
                os.getenv("S3_MODEL_PREFIX", "model")
            ),
            aws_region=os.getenv("AWS_REGION"),
            aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )

    @property
    def processed_s3_enabled(self) -> bool:
        return bool(self.s3_bucket_processed)

    @property
    def model_s3_enabled(self) -> bool:
        return bool(self.s3_bucket_model)


@dataclass(frozen=True)
class ForecastPipelineSettings:
    local_raw_dir: Path
    local_processed_dir: Path
    local_models_dir: Path
    s3_bucket_processed: str | None
    s3_processed_prefix: str
    aws_region: str | None
    aws_endpoint_url: str | None

    @classmethod
    def from_env(cls) -> "ForecastPipelineSettings":
        load_env_file()

        return cls(
            local_raw_dir=_resolve_project_path(
                os.getenv("RAW_LOCAL_DIR", "data/raw")
            ),
            local_processed_dir=_resolve_project_path(
                os.getenv("PROCESSED_LOCAL_DIR", "data/processed")
            ),
            local_models_dir=_resolve_project_path(
                os.getenv("MODELS_DIR", "models")
            ),
            s3_bucket_processed=(
                os.getenv("S3_BUCKET_PROCESSED")
                or os.getenv("S3_BUCKET_REFINED")
                or os.getenv("S3_BUCKET_RAW")
            ),
            s3_processed_prefix=_normalize_s3_prefix(
                os.getenv("S3_PROCESSED_PREFIX", "processed")
            ),
            aws_region=os.getenv("AWS_REGION"),
            aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )

    @property
    def processed_s3_enabled(self) -> bool:
        return bool(self.s3_bucket_processed)


@dataclass(frozen=True)
class AthenaCatalogSettings:
    aws_region: str | None
    aws_endpoint_url: str | None
    s3_bucket_raw: str | None
    s3_raw_prefix: str
    s3_bucket_refined: str | None
    s3_refined_prefix: str
    s3_bucket_processed: str | None
    s3_processed_prefix: str
    athena_database: str
    athena_workgroup: str
    athena_output_s3_uri: str | None

    @classmethod
    def from_env(cls) -> "AthenaCatalogSettings":
        load_env_file()

        return cls(
            aws_region=os.getenv("AWS_REGION"),
            aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            s3_bucket_raw=os.getenv("S3_BUCKET_RAW"),
            s3_raw_prefix=_normalize_s3_prefix(os.getenv("S3_RAW_PREFIX", "raw")),
            s3_bucket_refined=(
                os.getenv("S3_BUCKET_REFINED") or os.getenv("S3_BUCKET_RAW")
            ),
            s3_refined_prefix=_normalize_s3_prefix(
                os.getenv("S3_REFINED_PREFIX", "refined")
            ),
            s3_bucket_processed=(
                os.getenv("S3_BUCKET_PROCESSED")
                or os.getenv("S3_BUCKET_REFINED")
                or os.getenv("S3_BUCKET_RAW")
            ),
            s3_processed_prefix=_normalize_s3_prefix(
                os.getenv("S3_PROCESSED_PREFIX", "processed")
            ),
            athena_database=os.getenv("ATHENA_DATABASE", "tech_challenge_phase4"),
            athena_workgroup=os.getenv("ATHENA_WORKGROUP", "primary"),
            athena_output_s3_uri=os.getenv("ATHENA_OUTPUT_S3_URI"),
        )
