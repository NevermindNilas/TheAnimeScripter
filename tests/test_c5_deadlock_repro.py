"""
C5 deadlock reproduction.

Exercises the EXACT thread/queue/finally topology of main.py's standard pipeline
(`start()` -> `ThreadPoolExecutor(readBuffer, writeBuffer, process)`) and motionBlur's
(`run()` -> `ThreadPoolExecutor(readBuffer, writeBuffer, _processFrames)`).

Both share this structure:

    read-thread  ->  Queue(maxsize=32)  ->  process-thread  ->  Queue(maxsize=32)  ->  write-thread
                   (decodeBuffer)                                (writeBuffer)

The read-thread's `__call__` finally enqueues a None sentinel:
    finally:
        self.decodeBuffer.put(None)              # blocks if decodeBuffer is full

The process-thread's `finally` (baseline) only closes the write side:
    finally:
        self.writeBuffer.close()                 # enqueues None to writeBuffer

If processFrame raises mid-loop, the read-thread keeps producing. Once 32 frames
sit unconsumed in decodeBuffer, the read-thread's `decodeBuffer.put(frame)` blocks
forever, so its `finally` (which would enqueue the None sentinel) never runs.
`ThreadPoolExecutor.__exit__` -> `shutdown(wait=True)` waits on the read-thread
forever.

This file has two modes:
    BASELINE  = reproduce the bug (asserts that the executor hangs).
    FIXED     = assert the fixed drain pattern shuts down cleanly.

Run:
    python tests/test_c5_deadlock_repro.py baseline     # expect: HANG confirmed
    python tests/test_c5_deadlock_repro.py fixed        # expect: clean shutdown
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

DECODE_BUF_SIZE = 32  # mirrors ffmpegSettings.BuildBuffer.decodeBuffer
WRITE_BUF_SIZE = 32  # mirrors WriteBuffer.writeBuffer
TOTAL_FRAMES = 200  # >> DECODE_BUF_SIZE so the producer fills the queue
RAISE_AFTER = 5  # frames consumed before process() raises
SHUTDOWN_TIMEOUT = 8.0  # seconds; if executor not done by then => DEADLOCK


class FakeBuildBuffer:
    """Mirrors src/io/ffmpegSettings.py BuildBuffer public surface used by main.py."""

    def __init__(self):
        self.decodeBuffer = Queue(maxsize=DECODE_BUF_SIZE)
        self.isFinished = False
        self._frameAvailable = threading.Event()

    def __call__(self):
        # Mirrors BuildBuffer.__call__: emit frames, then in `finally` enqueue None.
        try:
            for i in range(TOTAL_FRAMES):
                self.decodeBuffer.put(i)  # BLOCKS when full
                self._frameAvailable.set()
        finally:
            self.decodeBuffer.put(None)  # baseline: blocks if full
            self.isFinished = True

    def read(self):
        return self.decodeBuffer.get()

    def isReadFinished(self):
        return self.isFinished


class FakeWriteBuffer:
    """Mirrors WriteBuffer public surface used by main.py."""

    def __init__(self):
        self.writeBuffer = Queue(maxsize=WRITE_BUF_SIZE)

    def __call__(self):
        # Mirrors WriteBuffer.__call__: drain until None sentinel.
        while True:
            frame = self.writeBuffer.get()
            if frame is None:
                break

    def write(self, frame):
        self.writeBuffer.put(frame)

    def close(self):
        self.writeBuffer.put(None)


def process_baseline(self, raise_after=RAISE_AFTER):
    """Verbatim copy of main.py:process() finally logic (baseline)."""
    frame_count = 0
    try:
        currentFrame = self.readBuffer.read()
        while currentFrame is not None:
            frame_count += 1
            if frame_count > raise_after:
                raise RuntimeError("simulated processFrame failure (e.g. CUDA OOM)")
            self.writeBuffer.write(currentFrame)
            currentFrame = self.readBuffer.read()
    except Exception as e:
        print(f"[process] exception: {e}", flush=True)
    finally:
        # BASELINE finally: only closes write side. read-thread left to starve.
        self.writeBuffer.close()
        print(
            "[process] finally: writeBuffer closed. readBuffer NOT drained.", flush=True
        )


def process_fixed(self, raise_after=RAISE_AFTER):
    """The fixed finally: also drain decodeBuffer so producer's put() unblocks."""
    frame_count = 0
    try:
        currentFrame = self.readBuffer.read()
        while currentFrame is not None:
            frame_count += 1
            if frame_count > raise_after:
                raise RuntimeError("simulated processFrame failure (e.g. CUDA OOM)")
            self.writeBuffer.write(currentFrame)
            currentFrame = self.readBuffer.read()
    except Exception as e:
        print(f"[process] exception: {e}", flush=True)
    finally:
        self.writeBuffer.close()
        print(
            "[process] finally: writeBuffer closed. Draining readBuffer...", flush=True
        )
        while not self.readBuffer.isReadFinished():
            try:
                self.readBuffer.decodeBuffer.get(timeout=0.1)
            except Empty:
                continue


class Pipeline:
    """Holds the three threads' state, mirrors VideoProcessor.start()."""

    def __init__(self, process_fn):
        self.readBuffer = FakeBuildBuffer()
        self.writeBuffer = FakeWriteBuffer()
        self._process = process_fn

    def run(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.readBuffer)
            executor.submit(self.writeBuffer)
            executor.submit(self._process, self)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    process_fn = {"baseline": process_baseline, "fixed": process_fixed}[mode]
    pipe = Pipeline(process_fn)

    done = threading.Event()

    def runner():
        pipe.run()
        done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    if done.wait(timeout=SHUTDOWN_TIMEOUT):
        print(
            f"\nRESULT ({mode}): executor shut down cleanly within "
            f"{SHUTDOWN_TIMEOUT}s — NO DEADLOCK."
        )
        sys.exit(0)
    else:
        print(
            f"\nRESULT ({mode}): executor still running after {SHUTDOWN_TIMEOUT}s "
            f"— DEADLOCK CONFIRMED. (In a real run, only SIGKILL or the "
            f"KeyboardInterrupt -> os._exit(130) bypass at main.py:804 escapes.)",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
