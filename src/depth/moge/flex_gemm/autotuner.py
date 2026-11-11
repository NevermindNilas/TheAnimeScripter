import builtins
from typing import *
import os
import json
import importlib
import pkgutil
import torch
import triton
import triton.testing
import time
import inspect
import functools
from filelock import FileLock

from . import config as pkg_config


_ADAPTIVE_NOTIFIED = set()


# Compact aliases for torch dtypes used in autotune cache keys / on-disk JSON.
# Keep STABLE: changing any existing entry invalidates previously persisted
# cache keys for that dtype. Add new dtypes; do not rename old ones.
_DTYPE_SHORT = {
    "torch.float16":  "f16",
    "torch.bfloat16": "bf16",
    "torch.float32":  "f32",
    "torch.float64":  "f64",
    "torch.int8":     "i8",
    "torch.int16":    "i16",
    "torch.int32":    "i32",
    "torch.int64":    "i64",
    "torch.uint8":    "u8",
    "torch.uint16":   "u16",
    "torch.uint32":   "u32",
    "torch.uint64":   "u64",
    "torch.bool":     "b",
}


def _short_dtype(dtype) -> str:
    s = str(dtype)
    return _DTYPE_SHORT.get(s, s)


def _notify_adaptive_tune(kernel_name):
    if kernel_name in _ADAPTIVE_NOTIFIED:
        return
    _ADAPTIVE_NOTIFIED.add(kernel_name)
    import sys
    print(
        f"FlexGEMM: autotune started for {kernel_name} after "
        f"{pkg_config.AUTOTUNE_ADAPTIVE_THRESHOLD} calls with the same key, this may take a while...",
        file=sys.stderr,
        flush=True,
    )


class TritonPersistentCacheAutotuner(triton.runtime.Autotuner):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )
        self._cache_key = _get_function_cache_key(fn)
        # Per-tuning-key call counts; adaptive mode triggers tuning once a
        # specific key has been observed `pkg_config.AUTOTUNE_ADAPTIVE_THRESHOLD` times.
        self._key_call_counts: Dict[str, int] = {}
        # Per-tuning-key timing metadata: {key: [{"config": str, "ms": float}, ...]}
        # Sorted ascending by ms; first entry is the autotune winner.
        # Used for offline analysis of pruning safety (how close are the
        # runner-ups to the winner). Persisted alongside `self.cache` but in
        # a sibling namespace so the existing on-disk format stays
        # backward-compatible.
        self.timings_meta: Dict[str, list] = {}
        _register_autotuner(self)

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(_short_dtype(arg.dtype))
            key = str(tuple(key))
            if key not in self.cache:
                mode = pkg_config.AUTOTUNE_MODE
                # Track per-key call count for adaptive mode. Only count
                # cache misses; once tuned, the key is in `self.cache` and
                # we never come back here.
                key_count = self._key_call_counts.get(key, 0) + 1
                self._key_call_counts[key] = key_count
                do_tune = (
                    mode == 'always'
                    or (mode == 'adaptive' and key_count >= pkg_config.AUTOTUNE_ADAPTIVE_THRESHOLD)
                )
                if not do_tune:
                    # Fall back to the first config without benchmarking and
                    # do not cache so we can re-evaluate once the threshold is
                    # crossed (adaptive mode) or the mode is changed.
                    config = self.configs[0]
                else:
                    # prune configs
                    used_cached_result = False
                    _notify_adaptive_tune(self.base_fn.__name__) if mode == 'adaptive' else None
                    pruned_configs = self.prune_configs(kwargs)
                    bench_start = time.time()
                    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    # Drop configs that failed to compile / run (Triton's
                    # Autotuner._bench swallows OutOfResources / PTXASError
                    # and returns inf-valued timings instead of raising). If
                    # any config worked, pick the best of those; if every
                    # config failed, surface a clear error pointing the user
                    # at the config list rather than letting the silently
                    # picked first config OOR at the actual call site.
                    working = {
                        cfg: t for cfg, t in timings.items()
                            if _bench_value_to_ms(t) != float('inf')
                    }
                    if not working:
                        failed = [_config_signature(c) for c in pruned_configs]
                        raise RuntimeError(
                            f"Autotune for {self.base_fn.__name__} key={key} found "
                            f"no viable config: all {len(failed)} candidates exceeded "
                            f"GPU resources (shared memory / registers). This typically "
                            f"means the configured tile sizes are too large for the "
                            f"current dtype / precision (e.g. `input_precision='ieee'` "
                            f"on fp32 uses ~2x the registers vs TF32). Add a smaller "
                            f"fallback config (smaller B1/B2/BK or num_stages=2) to "
                            f"the kernel's autotune list. Failed configs: {failed}"
                        )
                    self.cache[key] = builtins.min(working, key=working.get)
                    if pkg_config.AUTOTUNE_STORE_META:
                        self.timings_meta[key] = _summarize_timings(timings)
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                    self.configs_timings = timings
                    config = self.cache[key]
            else:
                config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__} finished after "
                  f"{self.bench_time:.2f}s; best config selected: {self.best_config};")
        if pkg_config.AUTOSAVE_AUTOTUNE_CACHE and not used_cached_result:
            save_autotune_cache()
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **config.all_kwargs(),
            ))
        self.nargs = None
        return ret


def triton_autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton_autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    """

    def decorator(fn):
        return TritonPersistentCacheAutotuner(
            fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
            post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
            use_cuda_graph=use_cuda_graph
        )

    return decorator


    





_AUTOTUNE_REGISTRY = {}
_PENDING_AUTOTUNE_CACHE = None

# Sibling namespace suffix appended to a kernel's cache_key when persisting
# per-tuning-key timing metadata (winner ms + top runner-ups). Kept separate
# from the primary config cache so the existing on-disk format keeps loading
# unchanged for older readers.
_META_KEY_SUFFIX = "::meta"
# How many top configs to record per tuning key (winner + runner-ups).
_META_TOP_K = 3


def _bench_value_to_ms(v):
    """Normalize the value returned by ``Autotuner._bench`` to a float (ms).
    Some Triton versions return a tuple (e.g. quantiles); take the first."""
    if isinstance(v, (tuple, list)):
        return float(v[0])
    return float(v)


def _config_signature(config):
    """Stable, JSON-safe string identifying a triton.Config for cache
    introspection."""
    try:
        return str(config)
    except Exception:
        return repr(config)


def _summarize_timings(timings, top_k: int = _META_TOP_K) -> list:
    """Return the top ``top_k`` configs from a ``{config: ms}`` timings dict,
    sorted ascending by ms. The first element is the autotune winner."""
    items = sorted(
        ((cfg, _bench_value_to_ms(v)) for cfg, v in timings.items()),
        key=lambda x: x[1],
    )[:top_k]
    return [{"config": _config_signature(cfg), "ms": round(ms, 3)} for cfg, ms in items]


def _unwrap_to_user_fn(fn):
    """Walk wrapper attributes (Heuristics.fn, JITFunction.fn, etc.) until we
    reach the innermost user-defined function. Returns the original ``fn`` if
    no wrapping is detected."""
    seen = set()
    current = fn
    while True:
        # Stop once we've reached a plain Python function defined in user code
        # (i.e. not living under triton.* or functools.*).
        mod = getattr(current, "__module__", None)
        if mod and not mod.startswith("triton.") and mod != "triton":
            return current
        inner = None
        for attr in ("fn", "f", "kernel", "_fn"):
            cand = getattr(current, attr, None)
            if cand is not None and id(cand) not in seen:
                inner = cand
                break
        if inner is None:
            return current
        seen.add(id(current))
        current = inner


def _get_callable_name(fn):
    fn = _unwrap_to_user_fn(fn)
    for attr in ("__name__", "__qualname__"):
        name = getattr(fn, attr, None)
        if name:
            return name
    return fn.__class__.__name__


def _get_function_cache_key(fn):
    fn = _unwrap_to_user_fn(fn)
    module = getattr(fn, "__module__", None) or getattr(fn.__class__, "__module__", "unknown")
    name = _get_callable_name(fn)
    return f"{module}.{name}"


def _get_device_name():
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name()
    except Exception:
        return None


def _get_cache_device_name(cache):
    device_name = _get_device_name()
    if device_name is None:
        return None
    if device_name in cache:
        return device_name
    if "*" in cache:
        return "*"
    return None


def _apply_cache_to_tuner(tuner, cache, device_name):
    cache_key = getattr(tuner, "_cache_key", None)
    if cache_key is None:
        return
    device_cache = cache.get(device_name, {})
    if cache_key in device_cache:
        cached_value = device_cache[cache_key]
        if isinstance(tuner, TritonPersistentCacheAutotuner):
            for k, v in cached_value.items():
                tuner.cache[k] = triton.runtime.Config(None)
                tuner.cache[k].__dict__.update(v)
    # Restore sibling meta (optional; absence is fine for legacy caches).
    meta_value = device_cache.get(cache_key + _META_KEY_SUFFIX)
    if meta_value and hasattr(tuner, "timings_meta"):
        tuner.timings_meta.update(meta_value)


def _register_autotuner(tuner):
    cache_key = getattr(tuner, "_cache_key", None)
    if cache_key is None:
        return
    _AUTOTUNE_REGISTRY[cache_key] = tuner
    if _PENDING_AUTOTUNE_CACHE:
        device_name = _get_cache_device_name(_PENDING_AUTOTUNE_CACHE)
        if device_name is not None:
            _apply_cache_to_tuner(tuner, _PENDING_AUTOTUNE_CACHE, device_name)
            

def get_autotune_cache():
    cache = {}
    device_name = _get_device_name()
    if device_name is None:
        return cache
    cache[device_name] = {}

    for cache_key, tuner in _AUTOTUNE_REGISTRY.items():
        if isinstance(tuner, TritonPersistentCacheAutotuner):
            cache[device_name][cache_key] = {k: v.__dict__ for k, v in tuner.cache.items()}
        if pkg_config.AUTOTUNE_STORE_META:
            meta = getattr(tuner, "timings_meta", None)
            if meta:
                cache[device_name][cache_key + _META_KEY_SUFFIX] = dict(meta)

    return cache


def save_autotune_cache(path=None):
    path = path or pkg_config.AUTOTUNE_CACHE_PATH
    lock_path = path + ".lock"

    with FileLock(lock_path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}
        # Merge existing cache with new cache
        cache.update(get_autotune_cache())

        tmp_path = path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(cache, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)


def load_autotune_cache(path_or_cache=None):
    cache = None

    # Preserve path-based loading, but allow callers to provide a preloaded cache object.
    if path_or_cache is None or isinstance(path_or_cache, (str, os.PathLike)):
        path = path_or_cache or pkg_config.AUTOTUNE_CACHE_PATH
        lock_path = path + ".lock"

        if not os.path.exists(path):
            return

        with FileLock(lock_path):
            with open(path, 'r') as f:
                cache = json.load(f)
    elif isinstance(path_or_cache, Mapping):
        cache = path_or_cache
    else:
        raise TypeError("load_autotune_cache expects a path or a mapping")

    if cache is None:
        return
    global _PENDING_AUTOTUNE_CACHE
    _PENDING_AUTOTUNE_CACHE = cache

    device_name = _get_cache_device_name(cache)
    if device_name is None:
        return

    for tuner in _AUTOTUNE_REGISTRY.values():
        _apply_cache_to_tuner(tuner, cache, device_name)
