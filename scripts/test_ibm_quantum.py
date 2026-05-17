"""
Tutorial script for testing IBM Quantum with a minimal experiment.

Goal
----
1. Run locally first by using an IBM fake backend.
2. Reuse almost the same code to run on real hardware.
3. Show the main concepts in the Qiskit Runtime flow:
   - quantum circuit
   - transpilation / ISA circuit
   - backend
   - SamplerV2 primitive
   - job mode
   - result inspection

How to use
----------
1) Install the dependencies:
   pip install "qiskit[all]~=2.3.1" "qiskit-ibm-runtime~=0.45.1" "qiskit-aer~=0.17"

2) Run a local test:
   python scripts/test_ibm_quantum.py --mode local

3) List real backends available to your account:
   python scripts/test_ibm_quantum.py --mode cloud --list-backends

4) Create or update the `.env` file at the project root:
   IBM_QUANTUM_API_TOKEN=YOUR_TOKEN
   IBM_QUANTUM_INSTANCE=YOUR_INSTANCE_OR_CRN

5) Run on real hardware:
   python scripts/test_ibm_quantum.py --mode cloud --backend ibm_brisbane --confirm-ibm-runtime-cost

Important notes
---------------
- This script uses job mode, which is compatible with Open Plan.
- It does not use Session because the official documentation states that
  Open Plan does not accept session jobs.
- In local mode, the script uses FakeManilaV2 to simulate an IBM backend.
- The chosen experiment is a Bell state because it is small and clearly shows
  entanglement: the expected dominant states are "00" and "11".
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _argv_requests_unconfirmed_cloud_run(argv: list[str]) -> bool:
    """Fast preflight before Qiskit imports to avoid accidental cloud jobs."""
    mode_is_cloud = "--mode=cloud" in argv
    for index, token in enumerate(argv[:-1]):
        if token == "--mode" and argv[index + 1] == "cloud":
            mode_is_cloud = True
            break

    return (
        mode_is_cloud
        and "--list-backends" not in argv
        and "--confirm-ibm-runtime-cost" not in argv
    )


if _argv_requests_unconfirmed_cloud_run(sys.argv[1:]):
    raise SystemExit(
        "Refusing to submit an IBM Quantum job without explicit confirmation.\n"
        "Cloud mode may consume IBM Quantum runtime minutes.\n"
        "Re-run with --confirm-ibm-runtime-cost after reviewing shots and backend."
    )


try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
except ImportError as exc:  # pragma: no cover - friendly setup message
    raise SystemExit(
        "Missing dependencies. Install them with:\n"
        'pip install "qiskit[all]~=2.3.1" "qiskit-ibm-runtime~=0.45.1" "qiskit-aer~=0.17"\n'
        f"\nTechnical details: {exc}"
    )


DEFAULT_SHOTS = 1024
DEFAULT_OPT_LEVEL = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass
class ExecutionConfig:
    mode: str
    backend_name: str | None
    shots: int
    optimization_level: int
    list_backends: bool
    confirm_ibm_runtime_cost: bool


def clean_env_value(value: str) -> str:
    """
    Remove wrapping quotes when the `.env` file stores quoted strings.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(env_path: Path = DEFAULT_ENV_PATH) -> bool:
    """
    Load a simple `.env` file by using only the standard library.

    Rules:
    - ignore comments and empty lines
    - accept an optional `export ` prefix
    - do not overwrite environment variables that are already defined
    """
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
        value = clean_env_value(value)

        if not key:
            continue

        os.environ.setdefault(key, value)

    return True


def parse_args() -> ExecutionConfig:
    parser = argparse.ArgumentParser(
        description="Minimal IBM Quantum test script with explanatory output."
    )
    parser.add_argument(
        "--mode",
        choices=("local", "cloud"),
        default="local",
        help="local uses a fake backend; cloud uses IBM Quantum Runtime.",
    )
    parser.add_argument(
        "--backend",
        dest="backend_name",
        default=None,
        help=(
            "Real backend name for cloud mode. "
            "If omitted, the script tries to choose the least_busy backend automatically."
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Number of shots for the SamplerV2 primitive.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=DEFAULT_OPT_LEVEL,
        choices=(0, 1, 2, 3),
        help="Transpilation optimization level.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="In cloud mode, list a few accessible real backends and exit.",
    )
    parser.add_argument(
        "--confirm-ibm-runtime-cost",
        action="store_true",
        help=(
            "Required with --mode cloud unless --list-backends is used. "
            "Confirms that this script may submit a job to IBM Quantum and "
            "consume runtime minutes."
        ),
    )
    args = parser.parse_args()
    return ExecutionConfig(
        mode=args.mode,
        backend_name=args.backend_name,
        shots=args.shots,
        optimization_level=args.optimization_level,
        list_backends=args.list_backends,
        confirm_ibm_runtime_cost=args.confirm_ibm_runtime_cost,
    )


def build_bell_circuit() -> QuantumCircuit:
    """
    Build a Bell circuit.

    Concept:
    - H on qubit 0 creates superposition.
    - CX between q0 and q1 creates entanglement.
    - measure_all adds classical registers and measures the qubits.
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def extract_counts(pub_result: Any) -> dict[str, int]:
    """
    Find the measurement BitArray and convert it into bitstring counts.

    In many examples from the documentation, when `measure_all()` is used,
    the default classical register name is `meas`. This helper still scans the
    available fields to make the lookup more robust.
    """
    data = pub_result.data

    if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
        return dict(data.meas.get_counts())

    for attr_name in dir(data):
        if attr_name.startswith("_"):
            continue
        attr_value = getattr(data, attr_name)
        if hasattr(attr_value, "get_counts"):
            return dict(attr_value.get_counts())

    raise RuntimeError(
        "Could not locate a classical register with get_counts() in the result."
    )


def print_counts(counts: dict[str, int], shots: int) -> None:
    print("\nResults (counts):")
    for bitstring, total in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        percentage = (total / shots) * 100
        print(f"  {bitstring}: {total} shots ({percentage:.2f}%)")


def build_cloud_service() -> QiskitRuntimeService:
    """
    Initialize the IBM Quantum Runtime service.

    Rules used here:
    - First, the script tries to load credentials from the `.env` file.
    - If a token is available in the environment, it instantiates the service explicitly.
    - If no token is available, it tries previously saved credentials.

    Environment variables expected by this script:
    - IBM_QUANTUM_API_TOKEN
    - IBM_QUANTUM_INSTANCE (recommended by the official documentation)
    """
    load_env_file()

    token = os.getenv("IBM_QUANTUM_API_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")

    if token:
        kwargs: dict[str, Any] = {
            "channel": "ibm_quantum_platform",
            "token": token,
        }
        if instance:
            kwargs["instance"] = instance
        return QiskitRuntimeService(**kwargs)

    return QiskitRuntimeService()


def choose_cloud_backend(
    service: QiskitRuntimeService,
    backend_name: str | None,
):
    """
    Choose the real backend.

    If the user provides `--backend`, use that backend name.
    Otherwise, use `least_busy` to find an operational backend.
    """
    if backend_name:
        return service.backend(backend_name)

    return service.least_busy(operational=True, simulator=False, min_num_qubits=2)


def list_cloud_backends(service: QiskitRuntimeService) -> None:
    backends = service.backends(operational=True, simulator=False)
    if not backends:
        print("No real backend was found for the current account.")
        return

    print("Accessible real backends:")
    for backend in backends[:15]:
        name = getattr(backend, "name", str(backend))
        qubits = getattr(backend, "num_qubits", "unknown")
        status = "operational"
        print(f"  - {name} | qubits={qubits} | status={status}")


def run_local(config: ExecutionConfig) -> None:
    print("Selected mode: local")
    print("Context: using FakeManilaV2 to validate the flow before the real QPU.")

    circuit = build_bell_circuit()
    backend = FakeManilaV2()

    print(f"Local backend: {backend.name}")
    print("Step: transpilation to adapt the circuit to the selected backend.")
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=config.optimization_level,
    )
    isa_circuit = pass_manager.run(circuit)

    print("Step: execution with SamplerV2 in job mode.")
    sampler = Sampler(
        mode=backend,
        options={"simulator": {"seed_simulator": 42}},
    )
    job = sampler.run([isa_circuit], shots=config.shots)
    result = job.result()
    counts = extract_counts(result[0])

    print("Final check: the dominant states should tend toward 00 and 11.")
    print_counts(counts, config.shots)


def run_cloud(config: ExecutionConfig) -> None:
    print("Selected mode: cloud")
    print("Context: using IBM Quantum Runtime with SamplerV2 in job mode.")
    if DEFAULT_ENV_PATH.exists():
        print(f".env file detected: {DEFAULT_ENV_PATH}")
    else:
        print(".env file not found; the script will try to use previously saved credentials.")

    try:
        service = build_cloud_service()
    except Exception as exc:
        raise SystemExit(
            "Could not authenticate with IBM Quantum Runtime.\n"
            "Set IBM_QUANTUM_API_TOKEN and preferably IBM_QUANTUM_INSTANCE in `.env`,\n"
            "or save your account with QiskitRuntimeService.save_account(...).\n"
            f"\nTechnical details: {exc}"
        )

    if config.list_backends:
        list_cloud_backends(service)
        return

    if not config.confirm_ibm_runtime_cost:
        raise SystemExit(
            "Refusing to submit an IBM Quantum job without explicit confirmation.\n"
            "Cloud mode may consume IBM Quantum runtime minutes.\n"
            "Re-run with --confirm-ibm-runtime-cost after reviewing shots and backend."
        )

    backend = choose_cloud_backend(service, config.backend_name)
    backend_name = getattr(backend, "name", str(backend))

    print(f"Selected real backend: {backend_name}")
    print("Step: build the Bell circuit.")
    circuit = build_bell_circuit()

    print("Step: local transpilation for the real backend ISA circuit.")
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=config.optimization_level,
    )
    isa_circuit = pass_manager.run(circuit)

    print("Step: submit the job to IBM Quantum Platform.")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=config.shots)
    print(f"Job ID: {job.job_id()}")
    print("Waiting for the remote result...")

    result = job.result()
    counts = extract_counts(result[0])

    print("Result received from the platform.")
    print_counts(counts, config.shots)


def main() -> int:
    config = parse_args()

    if config.mode == "local":
        run_local(config)
        return 0

    run_cloud(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
