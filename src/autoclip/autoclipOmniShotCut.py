import logging
import os

import torch

import src.constants as cs
from src.autoclip._frames import eightBit, openResizedReader, warnIfVariableRate
from src.autoclip._results import writeCutResults
from src.autoclip.omnishotcut_arch import (
    HEIGHT,
    INTER_NAMES,
    INTRA_NAMES,
    WIDTH,
    WINDOW,
)
from src.infra.logAndPrint import logAndPrint
from src.infra.progressBarLogic import ProgressBarLogic
from src.model.download import downloadModels
from src.model.registry import modelsMap, weightsDir

# Frames shared by adjacent windows (upstream inference.py's default); each
# window owns the cuts in its half of the overlap.
OVERLAP = 10
STRIDE = WINDOW - OVERLAP
_LEFT = OVERLAP // 2
_RIGHT = OVERLAP - _LEFT


def windowCuts(intra, inter, end, start, valid, last):
    """Global cuts one window owns, as ``(frame, intra, inter)``.

    ``intra``/``inter``/``end`` are the window's per-query argmax lists, in
    query order; ``start`` is the window's first global frame, ``valid`` its
    real (not black-padded) frame count, and ``last`` whether it reaches the
    end of the stream. Each label is the one of the shot that BEGINS at the
    cut. Mirrors upstream's ``decode_window_segments`` +
    ``collect_cuts_from_window``: a window's opening segment is skipped, since
    its neighbour sees that boundary as an interior one, except in the first
    window, where it comes out at frame 0 so ``cutFrames`` can tell that the
    stream opens inside a transition (upstream labels that shot General).
    """
    segments = []
    segStart = 0
    for intraLabel, interLabel, segEnd in zip(intra, inter, end, strict=True):
        segEnd = min(segEnd, valid)
        if segStart >= segEnd:
            continue  # degenerate or duplicate query
        segments.append((segStart, intraLabel, interLabel))
        segStart = segEnd
        if segEnd >= valid:
            break

    validStart = 0 if start == 0 else start + _LEFT
    validEnd = start + valid if last else start + WINDOW - _RIGHT
    cuts = [(0, *segments[0][1:])] if start == 0 and segments else []
    for local, intraLabel, interLabel in segments[1:]:
        frame = start + local
        if validStart < frame <= validEnd:
            cuts.append((frame, intraLabel, interLabel))
    return cuts


# Intra labels whose segment is a blend between two shots, and those that
# describe motion (wipe, push, slide, zoom, doorway). On anime the motion ones
# fire mostly on movement inside one shot (a door sliding shut, a character
# leaving to black, a camera pan), so they are not cuts.
_BLENDS = frozenset(INTRA_NAMES.index(name) for name in ("Dissolve", "Fade"))
_MOTION = frozenset(
    INTRA_NAMES.index(name) for name in ("Wipes", "Push", "Slide", "Zoom", "Doorway")
)
# Inter labels for a shot that opens with its own cut rather than out of the
# segment before it.
_OWN_CUT = frozenset(INTER_NAMES.index(name) for name in ("Hard_Cut", "Sudden_Jump"))


def cutFrames(cuts, total):
    """One cut frame per shot change, plus the deduplicated boundaries.

    The model splits a window into segments (``cuts`` are their starts, as
    ``(frame, intra, inter)``; frame 0 carries the opening segment's labels),
    and a transition is a segment of its own:

    * A run of blend segments (dissolve, fade; fade out then fade in is one
      run) is one cut at its midpoint, the edit point of a centred dissolve
      and where TransNetV2 cuts (within a frame of the middle of every dissolve
      on a test episode). A shot that opens with its own hard cut right after
      the run is a second cut, or the only one when the midpoint would leave a
      one-frame clip. A run at either end of the stream (an opening fade-in, a
      closing fade-out, a trim that starts inside a dissolve) belongs to the
      shot it fades into or out of.
    * A motion segment is not a cut, and the shot after it is one only when it
      opens with a hard cut or sudden jump. On a 10-minute anime episode,
      cutting at a motion segment's midpoint or start added six false cuts;
      this rule added none and missed three real ones.
    * Every other boundary is a cut.
    """
    kept = []
    for cut in sorted(cuts, key=lambda c: c[0]):
        if 0 <= cut[0] < total and (not kept or cut[0] != kept[-1][0]):
            kept.append(cut)
    opening = kept.pop(0) if kept and kept[0][0] == 0 else (0, 0, 0)

    frames = []
    blendStart = 0 if opening[1] in _BLENDS else None  # blend run in progress
    afterMotion = opening[1] in _MOTION
    for frame, intraLabel, interLabel in kept:
        if intraLabel in _BLENDS:
            if blendStart is None:
                blendStart = frame
            continue
        motion = intraLabel in _MOTION
        ownCut = not motion and interLabel in _OWN_CUT
        if blendStart is not None:
            middle = (blendStart + frame) // 2
            if blendStart > 0 and not (ownCut and frame - middle < 2):
                frames.append(middle)
            if ownCut:
                frames.append(frame)
            blendStart = None
        elif not motion and (not afterMotion or ownCut):
            frames.append(frame)
        afterMotion = motion
    return frames, kept


class AutoClipOmniShotCut:
    """
    Scene-change autoclip using OmniShotCut (UVA CV Lab, 2026).

    Streams nelux-decoded frames at the model's 128x96 in windows of 100 that
    overlap by 10, exactly as upstream's inference does (a last short window
    is padded with black frames). The model makes its own decisions (argmax
    over shot queries), so there is no threshold: ``--autoclip_sens`` has no
    effect. It runs in FP32 and decoding sets the pace (~1650 fps on a 1080p
    source on an RTX 3090, TransNetV2 ~1880; ~300 fps on the CPU at 720p).

    Memory bound is one window plus one frame (~3.7 MB of uint8).
    """

    def __init__(self, input, output, inPoint, outPoint):
        self.input = input
        self.output = output
        self.inPoint = inPoint
        self.outPoint = outPoint

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logAndPrint(
                "CUDA not available; running OmniShotCut on CPU. This will be "
                "significantly slower.",
                "yellow",
            )

        self._loadModel()

        if cs.ADOBE:
            from src.server.aeComms import progressState

            progressState.update({"status": "Detecting scene changes (omnishotcut)..."})
            try:
                self._run()
            except Exception as e:
                progressState.setFailed(error=str(e))
                raise
            progressState.setCompleted(outputPath=self.output)
        else:
            self._run()

    def _resolveModelPath(self):
        filename = modelsMap("omnishotcut")
        modelPath = os.path.join(weightsDir, "omnishotcut", filename)
        if not os.path.exists(modelPath):
            modelPath = downloadModels("omnishotcut")
        return modelPath

    def _loadModel(self):
        from src.autoclip.omnishotcut_arch import OmniShotCut

        modelPath = self._resolveModelPath()
        logging.info(f"Loading OmniShotCut weights from {modelPath}")
        self.model = OmniShotCut()
        state = torch.load(modelPath, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def _runWindow(self, window):
        """window: [WINDOW, H, W, 3] uint8 CPU -> three per-query lists."""
        batch = window.unsqueeze(0).to(self.device, non_blocking=True)
        intra, inter, end = (t[0].tolist() for t in self.model(batch))
        return intra, inter, end

    @staticmethod
    def _topUp(decoderIter, buf, want):
        """Decode until ``buf`` holds ``want`` frames or the stream ends."""
        frames = []
        while buf.shape[0] + len(frames) < want:
            try:
                frames.append(next(decoderIter))
            except StopIteration:
                break
        if not frames:
            return buf
        return torch.cat([buf, eightBit(torch.stack(frames, dim=0))], dim=0)

    def _run(self):
        # Upstream scales with ffmpeg's default (bicubic); nelux's bicubic
        # gives bit-identical frames, its default (bilinear) does not and
        # moved 10 of 157 cuts on a test episode.
        decoderIter, fps, startFrame, endFrame = openResizedReader(
            self.input,
            WIDTH,
            HEIGHT,
            self.inPoint,
            self.outPoint,
            resizeFilter="bicubic",
        )
        warnIfVariableRate(self.input, fps)
        startSec = startFrame / fps
        expected = endFrame - startFrame

        empty = torch.empty((0, HEIGHT, WIDTH, 3), dtype=torch.uint8)
        # One frame past the window tells whether this window is the last one
        # without trusting the container's frame count.
        buf = self._topUp(decoderIter, empty, WINDOW + 1)
        if buf.shape[0] == 0:
            raise RuntimeError(f"Decoded zero frames from {self.input}")

        cuts = []
        start = 0
        done = 0  # frames whose cuts are settled
        with ProgressBarLogic(expected, title="AutoClip (omnishotcut)") as pbar:
            while True:
                valid = min(WINDOW, buf.shape[0])
                last = buf.shape[0] <= WINDOW
                window = buf[:WINDOW]
                if valid < WINDOW:
                    pad = torch.zeros(
                        (WINDOW - valid, HEIGHT, WIDTH, 3), dtype=torch.uint8
                    )
                    window = torch.cat([window, pad], dim=0)

                intra, inter, end = self._runWindow(window)
                cuts.extend(windowCuts(intra, inter, end, start, valid, last))

                owned = start + valid if last else start + WINDOW - _RIGHT
                pbar.advance(owned - done)
                done = owned
                if last:
                    break
                buf = self._topUp(decoderIter, buf[STRIDE:], WINDOW + 1)
                start += STRIDE

            total = start + valid
            pbar.updateTotal(total)  # the container's count can be wrong

        frames, kept = cutFrames(cuts, total)
        for frame, intraLabel, interLabel in kept:
            logging.info(
                f"OmniShotCut boundary at frame {startFrame + frame}: "
                f"{INTRA_NAMES[intraLabel]} / {INTER_NAMES[interLabel]}"
            )
        writeCutResults([startSec + f / fps for f in frames], self.output)
