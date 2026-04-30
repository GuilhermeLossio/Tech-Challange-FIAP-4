# Update Note - 2026-04-27

## Topic
Initial read-only Flask client view for forecasts and training artifacts.

## Status
Completed for the first dashboard slice, then expanded with a six-month comparison view.

## Executive Summary
The repository now includes a separate Flask interface intended for the end-user client view. This surface is explicitly read-only and focuses on three product needs:

- inspect materialized future forecasts
- inspect training summaries for the classical and quantum models
- understand how prediction serving and data usage work without exposing any write path
- compare the standard and quantum forecast paths side by side on the home screen

## Recorded Decisions

### 1. Keep the client view read-only

The Flask interface exposes only `GET /` with filter query parameters. There are no upload, edit, retrain, or delete actions for standard users.

### 2. Preserve the serving constraint for quantum predictions

The dashboard can display `predict_type=quant` rows only when they were generated offline and materialized in `future_predict`. It never triggers IBM Quantum token usage at request time.

### 3. Separate live and batch prediction behavior in the UI

The dashboard shows:

- a live next-day prediction from the classical Keras model only
- a materialized forecast panel for `normal` and `quant` future rows

### 4. Provide a six-month context before showing the forecast comparison

The top of the dashboard now combines:

- the last six months of observed closes from the raw market dataset
- the standard future forecast trajectory
- the quantum future forecast trajectory

This makes the projected loss/gain scenarios readable against recent real price behavior instead of showing isolated future rows only.

This keeps the frontend aligned with the API contract already established in the project.

## Added Artifacts

- `src/front/app.py`
- `src/front/templates/dashboard.html`
- `src/front/static/app.css`
- `src/application/services/training_catalog_service.py`
- `scripts/run_front.py`

## Operational Notes

- The client view runs on Flask and defaults to `http://localhost:5001/`.
- The UI reads existing local artifacts only. It does not mutate `data/raw`, `data/processed`, or `models/`.
- The dashboard depends on already generated forecasts and already trained model manifests.
- Standard and quantum outlook cards expose upside, downside, horizon-end delta, and a short acquisition explanation for each serving path.

## Suggested Next Steps

1. Add symbol-level navigation and deeper drill-down pages.
2. Switch part of the dashboard to consume the FastAPI contract directly when the UI is deployed separately.
3. Add authentication and role separation before exposing the client view outside local environments.
