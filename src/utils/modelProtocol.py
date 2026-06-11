from typing import Any, Protocol

import torch


class ModelBackend(Protocol):
    """
    Structural interface shared by all single-model backends
    (upscale, interpolate, restore, segment, dedup, …).

    Depth backends load more than one model and use handleModels instead —
    see DepthBackend below.

    Convention (not enforced at runtime):
      - __init__ accepts the parsed args namespace and stores what it needs.
      - handleModel loads weights, builds the graph, and warms up the model.
      - __call__ processes one frame (or one frame-pair for interpolation).
        The exact extra arguments differ per capability; the first positional
        argument is always the current frame as a torch.Tensor.
    """

    def handleModel(self) -> None: ...

    def __call__(
        self, frame: torch.Tensor, /, *args: Any, **kwargs: Any
    ) -> torch.Tensor: ...


class DepthBackend(Protocol):
    """Structural interface for depth-estimation backends (multi-model load)."""

    def handleModels(self) -> None: ...

    def __call__(
        self, frame: torch.Tensor, /, *args: Any, **kwargs: Any
    ) -> torch.Tensor: ...
