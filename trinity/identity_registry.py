"""
Trinity identity resolver — the single indirection point for names/voices/personas
(identity-naming-layer-spec §3). Companion to model_registry.py.

Code asks for an `agent_id`; this returns its display_name / voice_id / persona_ref,
plus the system name and wake-word config. Display names live ONLY here, so a rename
is a registry edit, not a code change (spec §1). The persona name is a {{display_name}}
slot rendered at prompt-build time; voice-defining traits stay name-agnostic (spec §4)
so renaming changes what an agent is CALLED, not how it SOUNDS.

Target repo path: trinity/identity_registry.py
Reads:           trinity/config/identity-registry.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent / "config" / "identity-registry.yaml"


@dataclass(frozen=True)
class SystemIdentity:
    display_name: str
    wake_mode: str                  # typed | spoken
    wake_phrase: str
    enrollment_ref: str | None = None


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str                   # stable routing key — never renamed
    role: str
    display_name: str
    persona_ref: str
    voice_id: str | None = None
    enabled: bool = False


class IdentityRegistry:
    """Loads the identity registry and resolves agent_ids to their identity."""

    def __init__(self, path: Path = REGISTRY_PATH):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        sysblock = data.get("system", {}) or {}
        ww = sysblock.get("wake_word", {}) or {}
        if not sysblock.get("display_name"):
            raise ValueError(f"No system.display_name in identity registry at {path}")
        self._system = SystemIdentity(
            display_name=sysblock["display_name"],
            wake_mode=ww.get("mode", "typed"),
            wake_phrase=ww.get("phrase", ""),
            enrollment_ref=ww.get("enrollment_ref"),
        )

        agents = data.get("agents", []) or []
        if not agents:
            raise ValueError(f"No agents found in identity registry at {path}")
        self._agents: dict[str, AgentIdentity] = {}
        for a in agents:
            aid = a["agent_id"]
            self._agents[aid] = AgentIdentity(
                agent_id=aid,
                role=a.get("role", aid),
                display_name=a["display_name"],
                persona_ref=a.get("persona_ref", ""),
                voice_id=a.get("voice_id"),
                enabled=bool(a.get("enabled", False)),
            )

    # --- system-level reads ---------------------------------------------------
    def system(self) -> SystemIdentity:
        return self._system

    @property
    def system_name(self) -> str:
        return self._system.display_name

    @property
    def wake_phrase(self) -> str:
        return self._system.wake_phrase

    @property
    def wake_mode(self) -> str:
        return self._system.wake_mode

    # --- agent reads ----------------------------------------------------------
    def agent_ids(self, *, enabled_only: bool = False) -> list[str]:
        return sorted(a.agent_id for a in self._agents.values()
                      if a.enabled or not enabled_only)

    def agents(self, *, enabled_only: bool = False) -> list[AgentIdentity]:
        return [self._agents[i] for i in self.agent_ids(enabled_only=enabled_only)]

    def identity_for(self, agent_id: str) -> AgentIdentity:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(
                f"Unknown agent_id {agent_id!r}. Known: {sorted(self._agents)}"
            ) from None

    def display_name_for(self, agent_id: str) -> str:
        return self.identity_for(agent_id).display_name

    def voice_for(self, agent_id: str) -> str | None:
        return self.identity_for(agent_id).voice_id

    def persona_ref_for(self, agent_id: str) -> str:
        return self.identity_for(agent_id).persona_ref

    # --- persona rendering (spec §4) -----------------------------------------
    def render_persona(self, agent_id: str, template: str) -> str:
        """Interpolate the {{display_name}} / {{system_name}} slots in a persona
        template. ONLY the name-slot is substituted — voice-defining traits in the
        template must be written name-agnostic, so a rename never drifts the voice (§4)."""
        name = self.display_name_for(agent_id)
        return (template
                .replace("{{display_name}}", name)
                .replace("{{system_name}}", self.system_name))


# --- module-level convenience (lazy singleton) -------------------------------
_registry: IdentityRegistry | None = None


def registry() -> IdentityRegistry:
    global _registry
    if _registry is None:
        _registry = IdentityRegistry()
    return _registry


def display_name_for(agent_id: str) -> str:
    """Shorthand: trinity.identity_registry.display_name_for('orchestrator')."""
    return registry().display_name_for(agent_id)


def system_name() -> str:
    return registry().system_name


if __name__ == "__main__":
    reg = IdentityRegistry()
    s = reg.system()
    print(f"System: {s.display_name!r}   wake: mode={s.wake_mode} phrase={s.wake_phrase!r}\n")
    print("Roster (agent_id -> display_name [role]  voice  status):")
    for a in reg.agents():
        status = "live" if a.enabled else "planned"
        print(f"  {a.agent_id:14s} -> {a.display_name:8s} [{a.role:12s}] "
              f"voice={(a.voice_id or '<unset>'):14s} {status}")

    # Prove the rename-safe design: ONE name-agnostic template renders two different
    # display names with an IDENTICAL voice line. Renaming swaps the name, not the voice.
    template = (
        "You are {{display_name}}, part of the {{system_name}} system.\n"
        "  VOICE (fixed, name-agnostic): terse, factual, dry-witted."
    )
    for aid in ("orchestrator", "builder"):
        print(f"\nPersona render ({aid}):\n" + reg.render_persona(aid, template))
