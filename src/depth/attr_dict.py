from __future__ import annotations

from collections.abc import Mapping


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, AttrDict):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
        return value

    @classmethod
    def _unwrap(cls, value):
        if isinstance(value, AttrDict):
            return {key: cls._unwrap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._unwrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._unwrap(item) for item in value)
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def update(self, *args, **kwargs):
        for mapping in args:
            if mapping is None:
                continue
            if isinstance(mapping, Mapping):
                iterable = mapping.items()
            else:
                iterable = mapping
            for key, value in iterable:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def copy(self):
        return AttrDict(self)

    def to_dict(self):
        return self._unwrap(self)
