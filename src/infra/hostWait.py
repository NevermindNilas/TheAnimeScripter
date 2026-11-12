"""``--low_cpu``: sleep through most of a CUDA stream wait instead of spinning.

``cudaStreamSynchronize`` busy-waits under CUDA's default scheduling
(cudaDeviceScheduleAuto spins whenever the machine has more logical CPUs than
active contexts, i.e. always). Every stage waits for its stream once per frame,
so the processing thread keeps a whole core at 100% for the length of the run
while the GPU does the work: ~34 of the ~42 CPU seconds of a 720p shufflecugan
run.

Blocking sync (CU_CTX_SCHED_BLOCKING_SYNC) parks the thread instead, but every
wake then costs an interrupt and a deep C-state exit on the critical path:
-2.4% FPS on that run. This keeps the spin's wake latency: each call site
learns how long its waits take, sleeps most of that, and spins only the tail.
A nap that overshoots (the stream was already done on waking) halves the
estimate, so a site whose waits shrink falls back to spinning within a frame or
two.

It is opt-in because a thread that slept resumes on a cold core: the launch
work after each wait runs slower (~40% in isolation), which cost ~0.5% FPS on a
GPU-bound 1080p span run with cores to spare. Where the CPU is the limit (a
software encoder on few cores) the freed core is a net FPS gain.

``synchronize()`` keeps its contract: it returns only once ``query()``
(``cudaStreamQuery``) reports the stream finished or the original
``cudaStreamSynchronize`` returned, and anything ``query()`` raises hands the
wait to the original call, so errors surface exactly as they did.
"""

import sys
import time

# Sleep this share of the predicted wait at most; the rest is spun. Leaves
# room for per-frame variance in the GPU time of one call site.
_SLEEP_FRACTION = 0.9
# Wake at least this long before the predicted end: covers the OS timer's
# slack (Windows' high-resolution timer included) and a deep C-state exit
# (~1 ms on a 13700K's C3).
_WAKE_MARGIN_S = 0.004
# Shorter naps are not worth a timer round trip; such waits are spun.
_MIN_SLEEP_S = 0.001
# Weight of the newest wait in a site's running estimate.
_EMA_ALPHA = 0.2
# A nap that ended after the GPU finished cut into the next frame's launch.
_OVERSHOOT_BACKOFF = 0.5


class HybridWait:
    """Per call-site wait estimator; ``wait(key, query, spin)`` is one sync.

    ``query()`` says whether the stream finished, ``spin()`` is the original
    blocking synchronize. ``sleep``/``clock`` are injectable for tests.
    """

    def __init__(self, sleep=None, clock=None):
        self._sleep = sleep or time.sleep
        self._clock = clock or time.perf_counter
        self._estimates: dict = {}

    def wait(self, key, query, spin) -> None:
        if query():
            return
        estimate = self._estimates.get(key)
        start = self._clock()
        if estimate is not None:
            nap = min(estimate * _SLEEP_FRACTION, estimate - _WAKE_MARGIN_S)
            if nap >= _MIN_SLEEP_S:
                self._sleep(nap)
                if query():
                    self._estimates[key] = estimate * _OVERSHOOT_BACKOFF
                    return
        spin()
        waited = self._clock() - start
        self._estimates[key] = (
            waited if estimate is None else estimate + _EMA_ALPHA * (waited - estimate)
        )


class _QueryFailed(Exception):
    pass


def installHybridStreamWait(streamClass=None) -> bool:
    """Route ``streamClass.synchronize`` (``torch.cuda.Stream``) through a
    ``HybridWait``; False when it already was.

    The call site is the two frames above ``synchronize`` (the helper that
    syncs and whoever called it) plus the stream handle, so two stages sharing
    one sync helper, or one driver syncing a short copy and a long inference on
    one stream, keep separate estimates.
    """
    if streamClass is None:
        import torch

        streamClass = torch.cuda.Stream
    spin = streamClass.synchronize
    if getattr(spin, "_tasHybridWait", False):
        return False
    query = streamClass.query
    hybrid = HybridWait()
    getFrame = sys._getframe

    def synchronize(self) -> None:
        caller = getFrame(1)
        outer = caller.f_back
        key = (
            self.cuda_stream,
            caller.f_code,
            caller.f_lineno,
            outer.f_code if outer is not None else None,
            outer.f_lineno if outer is not None else 0,
        )

        def done() -> bool:
            try:
                return query(self)
            except Exception as e:
                raise _QueryFailed from e

        try:
            hybrid.wait(key, done, lambda: spin(self))
        except _QueryFailed:
            # Whatever query() tripped over (a sticky kernel error, a capture
            # in progress elsewhere) is the original call's to report.
            spin(self)

    synchronize.__doc__ = spin.__doc__
    synchronize._tasHybridWait = True
    streamClass.synchronize = synchronize
    return True


def enableLowCpuWait() -> bool:
    """``--low_cpu``: install the hybrid wait on NVIDIA CUDA; False elsewhere.

    ROCm reports HIP streams as ``torch.cuda`` but was not measured, and
    DirectML/OpenVINO/NCNN/MPS never wait on a CUDA stream.
    """
    import torch

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    installHybridStreamWait()
    return True
