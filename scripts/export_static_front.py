from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import warnings
from urllib.parse import parse_qsl


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*tf.function retracing.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.services.future_prediction_service import (  # noqa: E402
    FuturePredictionService,
)
from src.application.services.predictor_service import StandardPredictorService  # noqa: E402
from src.front.app import DEFAULT_HORIZON_DAYS, DEFAULT_SYMBOL, create_app  # noqa: E402
from src.infrastructure.config.settings import ForecastPipelineSettings  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "site"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the read-only Flask dashboard as static files that can be "
            "published to GitHub Pages."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the static site will be written. Defaults to ./site.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional symbol list. Defaults to all locally materialized symbols.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback value to render into each static page.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=DEFAULT_HORIZON_DAYS,
        help="Forecast horizon to render into each static page.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Forecast table row limit to render into each static page.",
    )
    parser.add_argument(
        "--extraction-date",
        default=None,
        help="Optional extraction_date partition in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir == PROJECT_ROOT:
        raise SystemExit("Refusing to export over the repository root.")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_static_assets(output_dir)

    settings = ForecastPipelineSettings.from_env()
    symbols = _resolve_symbols(settings=settings, requested_symbols=args.symbols)
    if not symbols:
        raise SystemExit(
            "No symbols were found. Generate or sync local data before exporting "
            "the static dashboard."
        )

    app = create_app()
    app.config["STATIC_EXPORT"] = True
    exported_paths: list[Path] = []
    with app.test_client() as client:
        for symbol in symbols:
            params = {
                "symbol": symbol,
                "predict_type": "all",
                "lookback": str(args.lookback),
                "horizon_days": str(args.horizon_days),
                "limit": str(args.limit),
            }
            if args.extraction_date:
                params["extraction_date"] = args.extraction_date

            response = client.get("/", query_string=params)
            if response.status_code != 200:
                raise SystemExit(
                    f"Failed to render {symbol}: HTTP {response.status_code}"
                )

            symbol_dir = output_dir / symbol.lower()
            symbol_dir.mkdir(parents=True, exist_ok=True)
            page_path = symbol_dir / "index.html"
            page_path.write_text(
                _rewrite_links(response.get_data(as_text=True), depth=1),
                encoding="utf-8",
            )
            exported_paths.append(page_path)

            if symbol == _select_index_symbol(symbols):
                index_html = _rewrite_links(response.get_data(as_text=True), depth=0)
                (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    _write_nojekyll(output_dir)
    print(f"Exported {len(exported_paths) + 1} HTML files to {output_dir}")


def _copy_static_assets(output_dir: Path) -> None:
    source_static = PROJECT_ROOT / "src" / "front" / "static"
    target_static = output_dir / "static"
    if target_static.exists():
        shutil.rmtree(target_static)
    shutil.copytree(source_static, target_static)


def _resolve_symbols(
    *,
    settings: ForecastPipelineSettings,
    requested_symbols: list[str] | None,
) -> tuple[str, ...]:
    if requested_symbols:
        normalized_symbols = {
            symbol.strip().upper()
            for symbol in requested_symbols
            if symbol.strip()
        }
        return tuple(sorted(normalized_symbols))

    predictor_service = StandardPredictorService(
        raw_root_dir=settings.local_raw_dir,
        processed_root_dir=settings.local_processed_dir,
        models_root_dir=settings.local_models_dir,
    )
    future_prediction_service = FuturePredictionService(
        processed_root_dir=settings.local_processed_dir,
    )
    return tuple(
        sorted(
            set(predictor_service.get_supported_symbols())
            | set(future_prediction_service.list_symbols())
        )
    )


def _select_index_symbol(symbols: tuple[str, ...]) -> str:
    if DEFAULT_SYMBOL in symbols:
        return DEFAULT_SYMBOL
    return symbols[0]


def _rewrite_links(html: str, *, depth: int) -> str:
    static_prefix = "../" * depth
    html = html.replace('href="/static/app.css"', f'href="{static_prefix}static/app.css"')
    html = html.replace('action="/"', 'action="../index.html"' if depth else 'action="index.html"')
    html = re.sub(
        r'href="/\?([^"]+)"',
        lambda match: f'href="{_query_to_static_href(match.group(1), depth=depth)}"',
        html,
    )
    html = html.replace('href="/"', 'href="../index.html"' if depth else 'href="index.html"')
    return html


def _query_to_static_href(query_string: str, *, depth: int) -> str:
    params = dict(parse_qsl(query_string, keep_blank_values=True))
    symbol = params.get("symbol", DEFAULT_SYMBOL).lower()
    prefix = "../" if depth else ""
    return f"{prefix}{symbol}/index.html"


def _write_nojekyll(output_dir: Path) -> None:
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
