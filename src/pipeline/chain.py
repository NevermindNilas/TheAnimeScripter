"""The processing chain the user typed: one ``StageSpec`` per stage, in order.

The CLI/JSON side (``src/cli/stageOrder.py``) builds ``args.chain`` from the
typed stage flags and their values; the engine side (``planner.planChain``,
``initializeModels``, ``executor.Pipeline``) consumes it. Nothing else in the
run decides what a stage does: per-stage settings live here, never on the
global ``args.<stage>_method``-style attributes.

Settings keys per kind (all resolved: defaults, inheritance and validation are
done before a spec is built, so every key is always present):

- ``dedup``: ``method`` (str), ``sens`` (float, the typed 0-100 value),
  ``threshold`` (float, ``sens`` mapped to the method's own scale).
- ``restore``: ``methods`` (tuple of str, run in sequence by one driver).
- ``upscale``: ``method`` (str), ``factor`` (int), ``customModel`` (str, "" if
  none).
- ``interpolate``: ``method`` (str), ``factor`` (float), ``ensemble`` (bool),
  ``dynamicScale`` (bool), ``staticStep`` (bool). The smooth-dedup instance
  (``smooth=True``) also has ``detector`` (str), ``sens`` (float),
  ``threshold`` (float) and ``maxSpan`` (int).
- ``resize``: ``target``, one of ``("size", width, height)``,
  ``("height", height)`` or ``("scale", factor)``. The planner turns it into
  concrete dimensions, because ``height`` and ``scale`` depend on the size the
  stage receives.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

KINDS = ("dedup", "restore", "upscale", "interpolate", "resize")


@dataclass(frozen=True)
class StageSpec:
    kind: str
    index: int  # 1-based occurrence of this kind in the chain
    settings: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    smooth: bool = False  # interpolate only: the --smooth_dedup instance

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown stage kind: {self.kind!r}")
        if self.index < 1:
            raise ValueError(f"stage index must be 1 or more, got {self.index}")
        if self.smooth and self.kind != "interpolate":
            raise ValueError("only an interpolate stage can be the smooth-dedup one")
        if not isinstance(self.settings, MappingProxyType):
            object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    @property
    def id(self) -> str:
        """Unique within a chain. The first stage of a kind keeps the bare kind
        name, so a chain without repeats has the same unit names as before."""
        return self.kind if self.index == 1 else f"{self.kind}#{self.index}"

    def __getitem__(self, key):
        return self.settings[key]


__all__ = ["KINDS", "StageSpec"]
