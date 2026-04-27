from __future__ import annotations

from collections import defaultdict
import threading
import time


class ApiMetricsService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._request_counters: dict[str, int] = defaultdict(int)
        self._error_counters: dict[str, int] = defaultdict(int)

    def record_request(self, route_name: str) -> None:
        with self._lock:
            self._request_counters[route_name] += 1

    def record_error(self, route_name: str) -> None:
        with self._lock:
            self._error_counters[route_name] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            request_lines = ["# TYPE api_requests_total counter"]
            error_lines = ["# TYPE api_request_errors_total counter"]
            total_requests = 0
            total_errors = 0

            for route_name in sorted(self._request_counters):
                value = self._request_counters[route_name]
                total_requests += value
                request_lines.append(
                    f'api_requests_total{{route="{route_name}"}} {value}'
                )

            for route_name in sorted(self._error_counters):
                value = self._error_counters[route_name]
                total_errors += value
                error_lines.append(
                    f'api_request_errors_total{{route="{route_name}"}} {value}'
                )

        uptime_seconds = time.monotonic() - self._started_at
        lines = [
            "# TYPE api_uptime_seconds gauge",
            f"api_uptime_seconds {uptime_seconds:.6f}",
            "# TYPE api_requests_total_sum counter",
            f"api_requests_total_sum {total_requests}",
            *request_lines,
            "# TYPE api_request_errors_total_sum counter",
            f"api_request_errors_total_sum {total_errors}",
            *error_lines,
            "# TYPE online_quantum_inference_enabled gauge",
            "online_quantum_inference_enabled 0",
        ]
        return "\n".join(lines) + "\n"

    def get_uptime_seconds(self) -> float:
        return time.monotonic() - self._started_at
