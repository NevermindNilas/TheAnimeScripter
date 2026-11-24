"""Interpolation backends.

Nothing is imported here on purpose: import the backend module you need
(``src.interpolate.rife``, ``src.interpolate.rife_ncnn``, ...), so each
optional ML dependency is loaded only by the backend that uses it and
``import src.interpolate`` works without torch.
"""
