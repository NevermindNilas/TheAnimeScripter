import os
import logging

import torch
import nelux

import src.constants as cs
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.logAndPrint import logAndPrint
from src.utils.progressBarLogic import ProgressBarLogic


class AutoClipTransnetv2:
    """
    Scene-change autoclip using TransNetV2 (Soucek & Lokoc, ACM MM 2024).

    Streams nelux-decoded frames at the model's native 48x27 resolution in
    non-overlapping STRIDE-sized chunks. A rolling buffer of WINDOW+STRIDE
    frames feeds the sliding-window inference (window=100, stride=50, keep
    middle 50). Per-frame logits go through sigmoid; rising-edge crossings
    of the threshold mark scene cuts.

    Memory bound is ~3 chunks (~600 KB at 48x27x3 uint8) regardless of video
    length — a 4-hour movie buffers no more than a 1-minute clip.
    """

    H, W = 27, 48
    WINDOW = 100
    STRIDE = 50
    PAD_LEFT = 25
    PAD_RIGHT_BASE = 25

    def __init__(self, input, threshold, inPoint, outPoint, half):
        self.input = input
        self.threshold = threshold
        self.inPoint = inPoint
        self.outPoint = outPoint
        self.half = half

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logAndPrint(
                "CUDA not available; running TransNetV2 on CPU. This will be "
                "significantly slower (~5-10x).",
                "yellow",
            )

        self._loadModel()
        self._run()

    def _resolveModelPath(self):
        filename = modelsMap("transnetv2")
        modelPath = os.path.join(weightsDir, "transnetv2", filename)
        if not os.path.exists(modelPath):
            modelPath = downloadModels("transnetv2")
        return modelPath

    def _loadModel(self):
        from src.autoclip.transnetv2_arch import TransNetV2

        modelPath = self._resolveModelPath()
        logging.info(f"Loading TransNetV2 weights from {modelPath}")
        # Drop the many_hot head: it doubles the dict-output path with no
        # autoclip use, and dict outputs complicate CUDA-graph capture.
        # strict=False so the unused cls_layer2.* keys in the checkpoint
        # are silently ignored.
        self.model = TransNetV2(use_many_hot_targets=False)
        state = torch.load(modelPath, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.model.to(self.device)
        # Note: channels_last_3d was tried and benchmarked ~10% slower for
        # this model. cuDNN's NDHWC Conv3d kernels aren't well-tuned for
        # TransNet's separable conv blocks at tiny spatial dims (27x48 down
        # to 3x6). Stay in NCHW.
        self._setupCudaGraph()

    def _setupCudaGraph(self):
        """Capture inference once with a static input tensor; replay per
        window. Falls back to plain forward if capture fails."""
        self.cudaGraph = None
        if self.device.type != "cuda":
            return
        try:
            self.staticInput = torch.zeros(
                (1, self.WINDOW, self.H, self.W, 3),
                dtype=torch.uint8,
                device=self.device,
            )
            self.cudaStream = torch.cuda.Stream()
            with torch.cuda.stream(self.cudaStream):
                with torch.inference_mode():
                    for _ in range(3):
                        warm = self.model(self.staticInput)
            self.cudaStream.synchronize()
            self.staticLogits = torch.empty_like(warm)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.cudaStream):
                with torch.inference_mode():
                    out = self.model(self.staticInput)
                    self.staticLogits.copy_(out)
            self.cudaGraph = graph
            logging.info("TransNetV2 CUDA graph captured")
        except Exception as e:
            logging.warning(
                f"CUDA graph capture failed for TransNetV2: {e}. "
                f"Falling back to eager forward."
            )
            self.cudaGraph = None

    def _openDecoder(self):
        # nelux iterator at 48x27 with prefetch is the fastest access pattern
        # available — get_batch_range is blocked when resize is configured.
        #
        # Do NOT switch this to ``reader.frame_at(i)`` in a loop: that path
        # calls seekToNearestKeyframe + decode-forward per call, giving
        # O(N^2)-ish behavior for sequential indices. ``__iter__`` (and the
        # range-slice form below) reuse the persistent decoder and stay O(1)
        # per frame.
        reader = nelux.VideoReader(
            self.input,
            backend="pytorch",
            decode_accelerator="cpu",
            resize=(self.W, self.H),
            num_threads=8,
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
        if endFrame <= startFrame:
            raise RuntimeError(
                f"Empty range: inPoint={self.inPoint}, outPoint={self.outPoint}"
            )

        if startFrame > 0 or endFrame < reader.frame_count:
            iterable = reader([startFrame / fps, endFrame / fps])
        else:
            iterable = reader

        try:
            reader.start_prefetch()
        except Exception:
            pass
        return iter(iterable), fps, startFrame, endFrame

    @torch.inference_mode()
    def _runWindow(self, window):
        """One inference call. window: [WINDOW, H, W, 3] uint8 CPU tensor."""
        batched = window.unsqueeze(0)
        if self.cudaGraph is not None:
            with torch.cuda.stream(self.cudaStream):
                self.staticInput.copy_(batched, non_blocking=True)
                self.cudaGraph.replay()
            self.cudaStream.synchronize()
            logits = self.staticLogits
        else:
            logits = self.model(batched.to(self.device, non_blocking=True))
        return torch.sigmoid(logits[0, :, 0])  # [WINDOW] on device

    def _decodeChunk(self, decoderIter, want):
        """Pull up to `want` frames from the nelux iterator and stack them.

        nelux 0.9.2+ allocates a fresh tensor per iteration (verified via
        ``data_ptr()``), so no clone is required. Older versions reused one
        underlying buffer and would have needed ``f.clone()`` here.
        """
        chunk = []
        for _ in range(want):
            try:
                chunk.append(next(decoderIter))
            except StopIteration:
                break
        if not chunk:
            return None
        return torch.stack(chunk, dim=0)

    def _run(self):
        decoderIter, fps, startFrame, endFrame = self._openDecoder()
        n = endFrame - startFrame
        startSec = startFrame / fps

        # Total padded frames so window math is clean. Mirror the original
        # TransNet padding recipe: PAD_LEFT at start, then enough at the end
        # for the last window to land on the final real frame's middle slice.
        rem = n % self.STRIDE
        padRight = self.PAD_RIGHT_BASE + (self.STRIDE - rem if rem else self.STRIDE)
        totalPadded = self.PAD_LEFT + n + padRight
        nWindows = (totalPadded - self.WINDOW) // self.STRIDE + 1

        # Pull the first real chunk so we have a first-frame template to
        # replicate for the PAD_LEFT prefix.
        firstChunk = self._decodeChunk(decoderIter, self.STRIDE)
        if firstChunk is None or firstChunk.shape[0] == 0:
            raise RuntimeError(f"Decoded zero frames from {self.input}")
        firstFrame = firstChunk[0:1]
        lastFrame = firstChunk[-1:]

        # Seed buffer with PAD_LEFT replicated copies of the first frame
        # followed by the first real chunk.
        buf = torch.cat(
            [firstFrame.expand(self.PAD_LEFT, -1, -1, -1), firstChunk], dim=0
        )
        decodedSoFar = firstChunk.shape[0]
        bufBase = 0  # padded index of buf[0]
        endOfStream = decodedSoFar >= n

        cutPredsCpu = []  # list of [STRIDE]-sized tensors (each on CPU)
        with ProgressBarLogic(n, title="AutoClip (transnetv2)") as pbar:
            for winIdx in range(nWindows):
                # Refill the buffer until it holds at least WINDOW frames.
                while buf.shape[0] < self.WINDOW:
                    if not endOfStream:
                        want = min(self.STRIDE, n - decodedSoFar)
                        chunk = self._decodeChunk(decoderIter, want)
                        if chunk is None or chunk.shape[0] == 0:
                            endOfStream = True
                            continue
                        decodedSoFar += chunk.shape[0]
                        if decodedSoFar >= n:
                            endOfStream = True
                        lastFrame = chunk[-1:]
                        buf = torch.cat([buf, chunk], dim=0)
                    else:
                        # Past end: replicate last decoded frame.
                        need = self.WINDOW - buf.shape[0]
                        buf = torch.cat(
                            [buf, lastFrame.expand(need, -1, -1, -1)], dim=0
                        )

                window = buf[: self.WINDOW]
                probs = self._runWindow(window)  # [WINDOW] on device
                middle = probs[self.PAD_LEFT : self.PAD_LEFT + self.STRIDE].cpu()
                cutPredsCpu.append(middle)

                # Real-frame progress: middle slice covers padded indices
                # [winIdx*STRIDE + PAD_LEFT, +STRIDE), which maps to real
                # frame indices [winIdx*STRIDE, +STRIDE) (clamped to [0, n]).
                realStart = winIdx * self.STRIDE
                realEnd = min(n, realStart + self.STRIDE)
                pbar.advance(max(0, realEnd - realStart))

                # Slide forward by STRIDE.
                buf = buf[self.STRIDE :]
                bufBase += self.STRIDE

                if endOfStream and bufBase >= self.PAD_LEFT + n:
                    break

        allPreds = torch.cat(cutPredsCpu, dim=0)[:n].numpy()

        cuts = []
        prevHigh = False
        for i, p in enumerate(allPreds):
            high = p > self.threshold
            if high and not prevHigh and i > 0:
                cuts.append(startSec + i / fps)
            prevHigh = high

        outPath = os.path.join(cs.WHEREAMIRUNFROM, "autoclipresults.txt")
        with open(outPath, "w") as f:
            for i, t in enumerate(cuts):
                logging.info(f"Scene {i + 1}: cut at {t:.3f}s")
                f.write(f"{t}\n")
        logAndPrint(f"AutoClip wrote {len(cuts)} cuts to {outPath}", "green")
