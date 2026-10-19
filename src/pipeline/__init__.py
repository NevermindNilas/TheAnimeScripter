"""The frame pipeline: planner (order -> chain) and executor (chain -> frames).

``planChain`` resolves the typed flag order into stages with their frame sizes.
``Pipeline`` runs those stages over the decoded stream. Import the submodules
directly; this package keeps no eager imports, so the CLI can load it without
torch.
"""
