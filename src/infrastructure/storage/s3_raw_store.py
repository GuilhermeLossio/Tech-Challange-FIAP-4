from __future__ import annotations

from pathlib import Path
from io import BytesIO

import pandas as pd


class S3RawStore:
    def __init__(
        self,
        bucket: str,
        prefix: str = "raw",
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("bucket must not be empty")

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "boto3 is required to upload raw data to S3. "
                "Install the dependencies from requirements.txt."
            ) from exc

        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3",
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

    def upload_dataframe(
        self,
        *,
        frame: pd.DataFrame,
        relative_path: Path,
    ) -> str:
        key = self.build_key(relative_path)
        buffer = BytesIO()
        try:
            frame.to_parquet(buffer, index=False)
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "A Parquet engine is required to upload raw data to S3. "
                "Install pyarrow from requirements.txt."
            ) from exc

        buffer.seek(0)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        return self.build_uri(key)

    def build_key(self, relative_path: Path) -> str:
        relative_value = relative_path.as_posix().lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{relative_value}"
        return relative_value

    def build_uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"
