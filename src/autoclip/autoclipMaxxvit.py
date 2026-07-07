import logging
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# nelux hard-requires torch to be imported first (checked at nelux import time),
# so torch MUST precede nelux here. Guard the order against isort reordering.
# isort: off
import torch  # noqa: F401
import nelux

# isort: on
import src.constants as cs
from src.infra.logAndPrint import logAndPrint
from src.infra.progressBarLogic import ProgressBarLogic
from src.sceneChange.scorer6ch import SceneChangeScorer6ch

_SENTINEL = object()


class AutoClipMaxxvit:
    """
    Scene-change autoclip backed by the MaxxVit ONNX model.

    Decoding runs in a dedicated thread via a ``ThreadPoolExecutor``; nelux
    decodes + downscales to 224x224 (libswscale) and pushes torch tensors
    (HWC uint8) into a bounded queue. The main thread runs preprocess +
    inference via the shared ``SceneChangeScorer6ch`` (the same 6-channel
    classifier used by the streaming interpolation-path detector). Output
    index ``[0][0]`` of the model's softmax is the cut probability; cut
    timestamps (seconds) are written to ``autoclipresults.txt``.
    """

    H = W = 224
    QUEUE_SIZE = 8

    def __init__(
        self,
        input,
        method,
        threshold,
        inPoint,
        outPoint,
        half,
    ):
        self.input = input
        self.method = method
        self.threshold = threshold
        self.inPoint = inPoint
        self.outPoint = outPoint
        self.half = half

        # Shared scoring core (model load + preprocess + score). Handles the
        # TRT-fp32 pin / DML session selection internally.
        self.scorer = SceneChangeScorer6ch(method, half, size=self.H)

        if cs.ADOBE:
            from src.server.aeComms import progressState

            progressState.update(
                {"status": f"Detecting scene changes ({self.method})..."}
            )
            try:
                self._run()
            except Exception as e:
                progressState.setFailed(error=str(e))
                raise
            outPath = os.path.join(cs.WHEREAMIRUNFROM, "autoclipresults.txt")
            progressState.setCompleted(outputPath=outPath)
        else:
            self._run()

    def _openReader(self):
        # Linear iter is the fast path. Do NOT switch to ``reader.frame_at(i)``
        # in a loop: that path seeks to the nearest keyframe + decodes forward
        # per call, giving O(N^2)-ish behavior. Range slicing via
        # ``reader([startSec, endSec])`` reuses the persistent decoder.
        # See autoclipTransnetv2._openDecoder for the full rationale: nelux
        # 0.10.x has an FF_THREAD_FRAME race when ``start_prefetch()`` is
        # called after construction (pthread_frame.c:174 async_lock
        # assertion). Construct with prefetch=True so the codec ctx is
        # bound to a single decode path from the start.
        reader = nelux.VideoReader(
            self.input,
            backend="pytorch",
            decode_accelerator="cpu",
            resize=(self.W, self.H),
            num_threads=8,
            prefetch=True,
        )
        fps = reader.fps
        if not fps or fps <= 0:
            raise RuntimeError(f"Invalid FPS reported for video: {self.input}")

        startFrame = int(round(float(self.inPoint) * fps)) if self.inPoint else 0
        endFrame = (
            int(round(float(self.outPoint) * fps))
            if self.outPoint
            else reader.frame_count
        )
        startSec = startFrame / fps
        totalFrames = max(1, endFrame - startFrame)

        if startFrame > 0 or endFrame < reader.frame_count:
            iterable = reader([startFrame / fps, endFrame / fps])
        else:
            iterable = reader

        return iterable, fps, startSec, totalFrames

    def _decodeWorker(self, iterable, preprocess, queue):
        # nelux 0.9.2+ allocates a fresh tensor per iteration (no buffer
        # reuse), so no clone is required. Running preprocess() here also
        # parallelizes cast/transpose with the main thread's inference.
        try:
            for frame in iterable:
                queue.put(preprocess(frame))
        finally:
            queue.put(_SENTINEL)

    def _run(self):
        iterable, fps, startSec, totalFrames = self._openReader()

        preprocess = self.scorer.preprocessHWC
        score = self.scorer.score

        queue: Queue = Queue(maxsize=self.QUEUE_SIZE)
        cuts = []
        prev = None
        idx = -1

        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="autoclip-decode"
        ) as pool:
            future = pool.submit(self._decodeWorker, iterable, preprocess, queue)
            try:
                with ProgressBarLogic(
                    totalFrames, title=f"AutoClip ({self.method})"
                ) as pbar:
                    while True:
                        curr = queue.get()
                        if curr is _SENTINEL:
                            break
                        idx += 1
                        pbar.advance(1)
                        if prev is None:
                            prev = curr
                            continue
                        if score(prev, curr) > self.threshold:
                            cuts.append(startSec + idx / fps)
                        prev = curr
                future.result()
            except BaseException:
                # Drain queue so the decoder thread can finish promptly.
                while True:
                    item = queue.get()
                    if item is _SENTINEL:
                        break
                raise

        outPath = os.path.join(cs.WHEREAMIRUNFROM, "autoclipresults.txt")
        with open(outPath, "w") as f:
            for i, t in enumerate(cuts):
                logging.info(f"Scene {i + 1}: cut at {t:.3f}s")
                f.write(f"{t}\n")
        logAndPrint(f"AutoClip wrote {len(cuts)} cuts to {outPath}", "green")
