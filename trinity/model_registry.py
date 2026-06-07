"""
Trinity model-role resolver — the single indirection point (FDS §4.2, pt 3).

Code asks for a ROLE; this returns the concrete (provider, model). Concrete model
IDs live only in model-registry.yaml, so a swap is a registry edit, not a code
change. Wiring the resolved values into Hermes's AIAgent happens at construction
(see `as_agent_kwargs`); the Phase-0 spike confirmed v0.16.0's AIAgent accepts
`provider` / `model` / `fallback_model`.

Target repo path: trinity/model_registry.py
Reads:           trinity/config/model-registry.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent / "config" / "model-registry.yaml"


@dataclass(frozen=True)
class ResolvedModel:
    role: str
    provider: str
    model: str
    escalation: str | None = None   # reserved hard-case model (e.g. Opus); not the default
    tier: int | None = None


class ModelRegistry:
    """Loads the role->model registry and resolves roles to concrete models."""

    def __init__(self, path: Path = REGISTRY_PATH):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._roles: dict = data.get("roles", {})
        if not self._roles:
            raise ValueError(f"No roles found in registry at {path}")

    def roles(self) -> list[str]:
        return sorted(self._roles)

    def model_for(self, role: str) -> ResolvedModel:
        """Resolve a role to its ACTIVE concrete model."""
        try:
            r = self._roles[role]
        except KeyError:
            raise KeyError(
                f"Unknown model role {role!r}. Known roles: {self.roles()}"
            ) from None
        return ResolvedModel(
            role=role,
            provider=r["provider"],
            model=r["model"],
            escalation=r.get("escalation"),
            tier=r.get("tier"),
        )

    def escalated(self, role: str) -> ResolvedModel:
        """The role's reserved escalation model (e.g. Opus for hard orchestration).
        Falls back to the base model if the role declares no escalation."""
        base = self.model_for(role)
        if not base.escalation:
            return base
        return ResolvedModel(
            role=role,
            provider=base.provider,      # same provider, heavier model
            model=base.escalation,
            escalation=None,
            tier=base.tier,
        )

    def as_agent_kwargs(self, role: str, *, hard: bool = False) -> dict:
        """Kwargs to splat into Hermes AIAgent construction.

        `hard=True` selects the role's reserved escalation model. Escalation is a
        *construction-time model choice* (the orchestrator decides a task is hard
        and builds with Opus) — it is deliberately NOT mapped to Hermes's
        `fallback_model`, which is error-resilience, not difficulty routing. Wire
        `fallback_model` separately if you want that behavior.

        NOTE: confirm exact kwarg names against runtime_provider.py before relying
        on this in production wiring.
        """
        rm = self.escalated(role) if hard else self.model_for(role)
        return {"provider": rm.provider, "model": rm.model}


# --- module-level convenience (lazy singleton) -------------------------------
_registry: ModelRegistry | None = None


def registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def model_for(role: str) -> ResolvedModel:
    """Shorthand: trinity.model_registry.model_for('orchestrator')."""
    return registry().model_for(role)


if __name__ == "__main__":
    # Quick smoke check: print the resolved active model for every role.
    reg = ModelRegistry()
    print(f"Registry roles: {reg.roles()}\n")
    for role in reg.roles():
        m = reg.model_for(role)
        line = f"  {role:20s} -> {m.provider}/{m.model}  (tier {m.tier})"
        if m.escalation:
            line += f"  [escalation: {m.escalation}]"
        print(line)
