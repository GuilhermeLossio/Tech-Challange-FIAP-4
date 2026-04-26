# Update Note - 2026-04-17

## Topic
Hybrid integration between the classical training pipeline and IBM Quantum.

## Status
Initial technical direction defined.

## Executive Summary
The initial direction is to evaluate a hybrid architecture that keeps the main data pipeline, preprocessing, feature engineering, and temporal training in the classical environment, while applying quantum experiments to smaller and more controlled blocks of the workflow.

## Context
The current project focuses on time-series forecasting for semiconductor assets with LSTM, historical market data, and news sentiment enrichment. In this setting, directly replacing the main temporal model with a quantum architecture is not the best first step.

## Recorded Decision
Adopt a hybrid strategy for exploring quantum computing with IBM Quantum.

## Proposed Technical Direction
- Keep ingestion, cleaning, normalization, feature fusion, and baseline temporal training in the classical environment.
- Reduce feature dimensionality before any quantum step, avoiding the direct submission of large temporal windows to the circuit.
- Start with lower-risk approaches, such as quantum kernels for movement-direction classification or D+1 return regression.
- Consider experiments with VQR and EstimatorQNN only after establishing a comparable classical baseline.
- Validate first in simulation or local testing mode, then evolve to execution on real IBM Quantum hardware.

## Rationale
- The current pipeline already separates preprocessing, feature fusion, and inference clearly.
- Financial time series with long windows are not a good first target for a full quantum replacement.
- A hybrid approach allows comparison of cost, predictive quality, and operational complexity with less risk.
- IBM Quantum can be explored incrementally without compromising the main project workflow.

## Suggested Next Steps
1. Define the first experimental target: direction classification or return regression.
2. Select a small set of derived window features.
3. Build a lightweight classical baseline for comparison.
4. Implement a first notebook or proof-of-concept module with Qiskit.
5. Measure the difference in performance, computational cost, and practical viability.

## Adopted Cataloging Pattern
- Folder: `Docs/`
- Filename: `YYYY-MM-DD-short-description.md`
- Minimum expected content:
  - topic
  - status
  - executive summary
  - recorded decision
  - rationale
  - next steps

## Note
This note introduces the pattern used to record technical project updates.
