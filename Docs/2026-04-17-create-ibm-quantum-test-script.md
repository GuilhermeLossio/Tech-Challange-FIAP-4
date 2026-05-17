# Update Note - 2026-04-17

## Topic
Creation of an educational script for the first IBM Quantum experiment.

## Status
Completed.

## Executive Summary
An introductory script was added to validate the Qiskit Runtime execution flow, first in local mode and then in cloud mode. The goal is to support gradual learning without forcing immediate execution on real hardware during the first contact with the platform.

## Recorded Decision
Adopt a smoke-test script with a Bell circuit and `SamplerV2`, prioritizing simplicity, conceptual clarity, and compatibility with `job mode`.

## Added Artifact
- `scripts/test_ibm_quantum.py`

## Rationale
- The Bell circuit is small, classical for first validation, and easy to interpret.
- Local mode reduces setup friction and validates the workflow before remote authentication.
- The use of `SamplerV2` follows the current IBM Quantum Runtime documentation model.
- The use of `job mode` avoids conflicts with the Open Plan limitation on `session jobs`.

## Educational Content Included In The Script
- difference between local mode and cloud mode
- purpose of transpilation
- role of the backend
- use of `SamplerV2`
- reading `counts`
- authentication through token and instance

## Suggested Next Steps
1. Run the script in local mode and validate the expected result.
2. Configure the IBM Quantum Platform token and instance.
3. List the backends accessible to the account.
4. Run the same experiment on real hardware with `--confirm-ibm-runtime-cost`.
5. Evolve from the Bell test to an experiment connected to project features.

## Note
This record complements the earlier hybrid integration note and documents the first practical artifact of that direction.
