from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import yaml


@dataclass
class ModelPricing:
    input: float
    cached_input: float
    output: float


@dataclass
class PricingTable:
    models: Dict[str, ModelPricing]


def load_pricing(path: str) -> PricingTable:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    models_raw = raw.get("models", {}) or {}
    models: Dict[str, ModelPricing] = {}
    for name, values in models_raw.items():
        if not isinstance(values, dict):
            continue
        try:
            models[name] = ModelPricing(
                input=float(values.get("input", 0.0)),
                cached_input=float(values.get("cached_input", 0.0)),
                output=float(values.get("output", 0.0)),
            )
        except (TypeError, ValueError):
            continue
    return PricingTable(models=models)


def resolve_pricing_path() -> str:
    override = os.environ.get("CHACK_PRICING")
    if override:
        return override
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "config", "pricing.yaml")


def estimate_cost(
    pricing: PricingTable,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> Optional[float]:
    if model not in pricing.models:
        return None
    rates = pricing.models[model]
    billable_prompt = max(prompt_tokens - cached_prompt_tokens, 0)
    total = (
        billable_prompt * rates.input
        + cached_prompt_tokens * rates.cached_input
        + completion_tokens * rates.output
    )
    return total / 1_000_000.0


def estimate_costs_by_model(
    pricing: PricingTable,
    usage_by_model: Dict[str, Tuple[int, int, int]],
) -> tuple[float, List[str]]:
    total = 0.0
    missing_models: List[str] = []
    for model_name, usage in usage_by_model.items():
        prompt_tokens, completion_tokens, cached_prompt_tokens = usage
        model_cost = estimate_cost(
            pricing,
            model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        if model_cost is None:
            missing_models.append(model_name)
            continue
        total += model_cost
    return total, missing_models
