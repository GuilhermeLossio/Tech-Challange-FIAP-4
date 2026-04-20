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
