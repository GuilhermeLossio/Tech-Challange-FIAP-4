from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill immutable and published model artifact references into "
            "existing Keras training manifests."
        )
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Root directory that contains manifests/, training_runs/, and published aliases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect manifests without writing the updated payloads to disk.",
    )
    return parser.parse_args()


def _resolve_candidate_immutable_path(
    *,
    history_path: Path,
    model_filename: str,
) -> Path:
    return (history_path.parent / model_filename).resolve()


def main() -> int:
    args = _parse_args()
    models_dir = Path(args.models_dir).expanduser().resolve()
    manifests_root = models_dir / "manifests"
    if not manifests_root.exists():
        raise FileNotFoundError(f"Manifest root not found: {manifests_root}")

    counters = Counter()
    changed_manifests = 0

    for manifest_path in sorted(
        manifests_root.glob("extraction_date=*/trained_at=*/keras_training_manifest.json")
    ):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_changed = False

        for asset in payload.get("assets", []):
            counters["assets_total"] += 1
            model_path = Path(str(asset.get("model_local_path", ""))).expanduser().resolve()
            history_path = Path(str(asset.get("history_local_path", ""))).expanduser().resolve()
            model_filename = model_path.name
            published_alias_path = (models_dir / model_filename).resolve()
            immutable_candidate = _resolve_candidate_immutable_path(
                history_path=history_path,
                model_filename=model_filename,
            )

            immutable_model_local_path: str | None = None
            if "training_runs" in model_path.parts and model_path.exists():
                immutable_model_local_path = str(model_path)
                counters["immutable_existing"] += 1
            elif immutable_candidate.exists():
                immutable_model_local_path = str(immutable_candidate)
                counters["immutable_backfilled"] += 1
            else:
                counters["legacy_alias_only"] += 1

            published_model_local_path = (
                str(published_alias_path) if published_alias_path.exists() else None
            )
            artifact_reference_mode = (
                "immutable" if immutable_model_local_path is not None else "legacy_alias"
            )

            updates = {
                "immutable_model_local_path": immutable_model_local_path,
                "published_model_local_path": published_model_local_path,
                "artifact_reference_mode": artifact_reference_mode,
            }
            for key, value in updates.items():
                if asset.get(key) != value:
                    asset[key] = value
                    manifest_changed = True

        if manifest_changed:
            changed_manifests += 1
            counters["manifests_changed"] += 1
            if not args.dry_run:
                manifest_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

    counters["manifests_scanned"] = sum(
        1
        for _ in manifests_root.glob(
            "extraction_date=*/trained_at=*/keras_training_manifest.json"
        )
    )

    print(f"models_dir={models_dir}")
    print(f"dry_run={args.dry_run}")
    print(f"changed_manifests={changed_manifests}")
    for key in sorted(counters):
        print(f"{key}={counters[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
