from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MethodCatalogItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method_id: str
    availability: str
    predict_type: str
    title: str
    summary: str
    inputs: list[str]
    outputs: list[str]
    limitations: list[str]


class MethodCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    online_quantum_inference_enabled: bool
    methods: list[MethodCatalogItemResponse]
    supported_symbols: list[str]
    latest_extraction_date: str
