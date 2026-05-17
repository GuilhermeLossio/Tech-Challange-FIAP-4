# Update Note - 2026-05-16

## Topic
Quantum simulator and real IBM Quantum execution strategy for the model comparison pipeline.

## Status
Technical direction adopted for documentation and future implementation.

## Executive Summary
The project should compare three execution paths: the classical Keras LSTM baseline, the Qiskit quantum simulator path, and a tightly budgeted IBM Quantum hardware path. The quantum model should receive normalized, compressed, low-dimensional features rather than raw price windows. Real hardware execution must remain offline and explicit because each optimizer evaluation can consume runtime jobs, shots, queue time, and IBM Quantum minutes.

## Context
The current training comparison already separates the classical and quantum workflows:

- `KerasTrainingService` trains a classical LSTM for next-day closing-price regression.
- `TrainQuantumModelUseCase` trains a hybrid VQC for next-day direction classification.
- `scripts/train_and_compare_models.py` compares training time and directional metrics.
- `GenerateForecastBatchUseCase` materializes quantum predictions offline so the API never triggers live quantum inference.

This structure is appropriate for the project objective: compare predictive quality, execution time, operational cost, and practical limits of a quantum approach against a standard classical baseline.

## Recorded Decision
Adopt a three-layer benchmark:

| Layer | Purpose | IBM Quantum cost | Recommended use |
|---|---|---:|---|
| Classical Keras LSTM | Production baseline and price regression reference | none | Main baseline |
| Qiskit simulator | Quantum pipeline validation without cloud cost | none | Default quantum experiment |
| IBM Quantum hardware | Real-noise and real-runtime experiment | consumes minutes | Small, explicit, offline runs |

The simulator path should be the safe default. Hardware execution should be treated as a controlled experiment, not as the default training path.

## Normalization Strategy
The quantum circuit should not receive raw 60-day price windows directly. The recommended quantum feature flow is:

1. Build compact financial features from the historical window.
2. Standardize features with `StandardScaler`.
3. Reduce dimensionality with `PCA` to match the selected number of qubits.
4. Scale reduced values into `[0, pi]` for stable angle encoding.
5. Encode the values with `ZZFeatureMap`.
6. Train the variational classifier with `RealAmplitudes` and a classical optimizer.

This matches the current implementation direction in `TrainQuantumModelUseCase._build_quantum_features`.

## Real Hardware Guardrails
When moving to IBM Quantum hardware, use a deliberately small configuration:

```bash
python scripts/train_and_compare_models.py \
  --symbols NVDA \
  --quantum-mode cloud \
  --quantum-shots 256 \
  --quantum-optimizer spsa \
  --quantum-optimizer-maxiter 15 \
  --quantum-max-train-samples 24 \
  --quantum-max-validation-samples 16 \
  --quantum-max-test-samples 16 \
  --confirm-ibm-runtime-cost \
  --skip-s3
```

Recommended hardware defaults:

| Parameter | Recommended range | Reason |
|---|---:|---|
| `quantum_num_qubits` | 2 to 3 | Keeps circuit width small |
| `quantum_feature_map_reps` | 1 | Reduces depth and noise exposure |
| `quantum_ansatz_reps` | 1 | Reduces trainable parameters |
| `quantum_shots` | 128 to 256 | Controls runtime cost |
| `quantum_optimizer` | `spsa` | Designed for noisy objective functions |
| `quantum_optimizer_maxiter` | 10 to 20 | Limits function evaluations |
| `quantum_max_train_samples` | 16 to 32 | Limits circuit executions |

## Reporting Requirements
The final comparison report should make execution cost and environment visible:

- execution mode: `classical`, `simulator`, or `cloud`
- backend name
- shots
- optimizer name
- optimizer max iterations
- function evaluations
- training sample count
- validation sample count
- test sample count
- wall-clock training time
- directional accuracy, precision, recall, and F1
- whether IBM Quantum runtime minutes were consumed

This makes the benchmark honest: the quantum path does not need to outperform the LSTM to be valuable. It needs to be measured transparently.

## Circuit Diagrams
The diagrams added with this note document the circuit-level plan:

- [VQC circuit architecture](graphs/quantum_vqc_circuit_architecture.svg)
- [Quantum hardware execution loop](graphs/quantum_hardware_execution_loop.svg)

## Suggested Next Steps
1. Extend the comparison manifest with a `cost_profile` block.
2. Add a report section named "Execution Environment and Cost".
3. Keep `local` simulator mode as the default path for demos and repeated tests.
4. Use cloud mode only for small single-symbol benchmark runs.
5. Keep `--confirm-ibm-runtime-cost` mandatory for any job-submitting cloud command.
