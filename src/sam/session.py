"""``--sam_session``: click-to-track for a front end, over stdin and stdout.

The front end writes one JSON request per line and reads lines starting with
``PREFIX``. A request is ``{"id": 1, "cmd": "open", ...}`` and is answered by
``{"id": 1, "ok": true, ...}`` or ``{"id": 1, "ok": false, "error": "..."}``. While a
request runs, lines without an id report on it: ``{"event": "status", "message": ...}``
and ``{"event": "progress", "done": n, "total": m}``. ``{"cmd": "cancel"}`` is handled
as soon as it arrives and stops a track that is running or already requested; with
neither, it does nothing.

Commands:

``open``    ``{video, model}``. Loads the model and the video.
``prompt``  ``{frame, object, points: [{x, y, positive}]}``. Replaces that object's
            points on that frame and answers with a preview of every object there.
``frame``   ``{frame}``. Answers with that decoded frame: ``{jpeg, width, height}``,
            ``jpeg`` being base64. Leaves the tracking state alone.
``remove``  ``{object}``. Forgets an object and answers with ``previews``, the
            ``{frame, mask}`` of every frame that still has clicks.
``track``   ``{start, end, output}``. Follows every object over source frames
            ``start..end`` and writes one matte per frame, which
            ``--segment_mask`` reads back as a folder. With ``output`` null it
            tracks the same way and writes nothing, which is how a front end
            replays ``session.json``'s ``ops``. ``direction`` "forward" or
            "backward" tracks from frame ``from`` only, on top of a copy of the
            earlier track in ``base``. ``stopAtCuts`` stops at scene cuts; a
            later shot is then tracked only from a click inside it.
``trackShape`` ``{frame, to, points, stopAtCuts}``. Follows ``[x, y]`` points with
            optical flow and answers with ``frames``, the points on every frame up
            to ``to`` (or ``stoppedAt``, the scene cut it stopped before).
``composite`` ``{frame, doc, live, labels}``. Answers with ``matte``, the final matte of
            that frame for a roto document (src/segment/rotoDoc.py) as a base64
            PNG at preview size. ``live`` reads the frame from the clicks rather
            than from the document's mattes. With ``labels`` it adds ``labels``, a
            PNG of which object each pixel belongs to (0 for none).
``quit``    Ends the session.

A matte, and a click's preview, is white where an object is and black elsewhere.
"""

import base64
import gc
import json
import logging
import os
import queue
import shutil
import statistics
import sys
import threading
import traceback
import zlib
from collections import OrderedDict

import cv2
import numpy as np

from src.masking import listMaskFrames

PREFIX = "@@TAS-SAM "
SESSION_FILE = "session.json"

_IMAGE_SIZE = 1024
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)
_PREVIEW_MAX_SIDE = 1280
_DECODE_CHUNK = 8
# How far back to start decoding when a frame cannot be reached directly.
_WALK_BACK = (16, 64, 256, 1024)
_OUTPUT_STORES = ("cond", "nonCond", "tempCond", "tempNonCond")
_FRAME_CACHE = 24
# Upstream reads at most 16 frames back for object pointers and 7 for memory.
_KEEP_TRACKED_FRAMES = 20
# Scene cuts: mean absolute difference of 16x16 thumbnails of the normalized frames,
# against the differences beside it (see SamSession._isCut).
_CUT_GRID = 16
_CUT_THRESHOLD = 0.3
_CUT_RATIO = 3.0
_CUT_FLOOR = 0.02
_CUT_REACH = 4
_CUT_SIGNATURES = 64
# Beside each matte: which object each pixel belongs to, for the front end.
LABELS_DIR = "labels"
# ``composite`` was named a document version it does not have: send the document.
ROTO_DOC_MISSING = "roto-doc-missing"
_MAX_SHAPE_POINTS = 4096
# What a ``cancel`` stops.
_CANCELLABLE = ("track", "trackShape")
# What a track changes in the predictor's state, and how many levels of dict sit
# above the per-frame values.
_TRACKING_STATE = {
    "output_dict_per_obj": 3,
    "temp_output_dict_per_obj": 3,
    "frames_tracked_per_obj": 2,
}
# What a click or a removal can change besides.
_REQUEST_STATE = {
    **_TRACKING_STATE,
    # One deeper: confirming a click writes its memory into the output itself.
    "output_dict_per_obj": 4,
    "temp_output_dict_per_obj": 4,
    "point_inputs_per_obj": 2,
    "mask_inputs_per_obj": 2,
    "obj_id_to_idx": 1,
    "obj_idx_to_id": 1,
    "obj_ids": 1,
}


class Cancelled(Exception):
    pass


def maskFileName(frame: int) -> str:
    return f"{frame:06d}.png"


def edgeCoverage(logits):
    """Object coverage in 0..1 from a [H, W] logit map, anti-aliased over one pixel.

    The object is where the logit is positive, as in the previews. Each pixel's
    signed distance to that boundary is estimated as logit / |gradient|, so the
    edge gets a one pixel ramp at its true sub-pixel position while the inside
    stays fully opaque. The model's sigmoid would instead leave low-confidence
    parts of the object partly see-through.
    """
    import torch

    dy, dx = torch.gradient(logits)
    slope = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
    return (0.5 + logits / slope).clamp_(0.0, 1.0)


def matteImage(coverage: np.ndarray) -> np.ndarray:
    """Object coverage in 0..1 as an 8-bit ``--segment_mask`` frame, white on the object."""
    return np.rint(np.clip(coverage, 0.0, 1.0) * 255.0).astype(np.uint8)


def previewPng(mask: np.ndarray) -> str:
    """A small base64 PNG of a boolean mask, for showing a click's result."""
    height, width = mask.shape
    scale = min(1.0, _PREVIEW_MAX_SIDE / max(height, width))
    image = mask.astype(np.uint8) * 255
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_NEAREST,
        )
    encoded, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not encoded:
        raise RuntimeError("Could not encode the preview.")
    return base64.b64encode(data.tobytes()).decode("ascii")


class FrameSource:
    """Decoded, preprocessed frames by index, for the predictor.

    Tracking walks the video one frame at a time in either direction, so a miss
    decodes a short run ahead in the direction of travel. Only a window stays in
    memory, so the length of the video does not matter.
    """

    def __init__(self, path: str):
        import nelux

        self.path = path
        self.reader = nelux.VideoReader(path)
        self._fresh = True
        self._frameIndex = None
        try:
            claimed = int(self.reader.frame_count)
        except RuntimeError as error:
            # Counting walks the packets, so a damaged file fails here. Random
            # access fails with it, and tracking needs that, so there is no
            # readable part of the file to fall back on.
            raise ValueError(f"This video is damaged: {error}") from error
        self._frameIndex = self._readerFrameIndex()
        self.count = self._decodedCount(claimed)
        self.width = int(self.reader.width)
        self.height = int(self.reader.height)
        self.fps = float(self.reader.fps)
        self._cache: OrderedDict = OrderedDict()
        self._last = None

    def __len__(self) -> int:
        return self.count

    def _readerFrameIndex(self):
        """The reader's packet index, so that replacing the reader need not rescan."""
        try:
            return self.reader.frame_index
        except AttributeError, RuntimeError:
            return None

    def _reopen(self) -> None:
        import nelux

        close = getattr(self.reader, "close", None)
        if close is not None:
            try:
                close()
            except RuntimeError:
                logging.exception("Could not close a video reader")
        if self._frameIndex is None:
            self.reader = nelux.VideoReader(self.path)
        else:
            self.reader = nelux.VideoReader(self.path, frame_index=self._frameIndex)
        self._fresh = True

    def _attempt(self, read):
        """The frames ``read`` returns, or None if decoding them failed."""
        self._fresh = False
        try:
            frames = read()
        except Exception as error:
            logging.info(f"A video read failed: {error}")
            return None
        return frames if len(frames) > 0 else None

    def _walkTo(self, earliest: int, start: int, end: int):
        """Frames ``start..end-1``, decoded forward in short reads from ``earliest``."""
        import torch

        kept = []
        cursor = earliest
        while cursor < end:
            batch = self.reader.get_batch_range(
                cursor, min(cursor + _DECODE_CHUNK, end)
            )
            if len(batch) == 0:
                raise RuntimeError("The decoder returned no frames.")
            if cursor + len(batch) > start:
                kept.append(batch[max(0, start - cursor) :])
            cursor += len(batch)
        return torch.cat(kept)

    def _readRange(self, start: int, end: int):
        """Frames ``start..end-1``, however the decoder has to be asked for them.

        A reader that has seeked can be left broken for good (a backward seek in
        MPEG-TS with B-frames does it), so a failure there gets one try on a new
        reader. A new reader can still refuse the last group of pictures of a TS
        file, while decoding into it from the group before works.
        """
        wasFresh = self._fresh
        frames = self._attempt(lambda: self.reader.get_batch_range(start, end))
        if frames is None and not wasFresh:
            self._reopen()
            frames = self._attempt(lambda: self.reader.get_batch_range(start, end))
        earliest = start
        for reach in _WALK_BACK:
            if frames is not None or earliest == 0:
                break
            earliest = max(0, start - reach)
            self._reopen()
            frames = self._attempt(
                lambda earliest=earliest: self._walkTo(earliest, start, end)
            )
        if frames is None:
            self._reopen()
            which = (
                f"Frame {start}" if end - start == 1 else f"Frames {start}-{end - 1}"
            )
            raise ValueError(f"{which} of this video could not be decoded.")
        return frames

    def _decodes(self, index: int) -> bool:
        try:
            self._readRange(index, index + 1)
        except ValueError:
            return False
        return True

    def _decodedCount(self, claimed: int) -> int:
        """How many frames really decode, given how many the reader claims.

        The claim can run past the last frame that decodes. Everything after the
        end fails alike, so one read proves the claim right and a bisection finds
        the real end when it is wrong.
        """
        if claimed <= 0 or self._decodes(claimed - 1):
            return max(claimed, 0)
        low, high = -1, claimed - 1
        while high - low > 1:
            middle = (low + high) // 2
            if self._decodes(middle):
                low = middle
            else:
                high = middle
        logging.warning(f"The video claims {claimed} frames but decodes {low + 1}.")
        return low + 1

    def _decode(self, start: int, end: int) -> None:
        import torch
        import torch.nn.functional as F

        batch = self._readRange(start, end)
        scale = 65535.0 if batch.dtype == torch.uint16 else 255.0
        images = batch.permute(0, 3, 1, 2).to(torch.float32).div_(scale)
        images = F.interpolate(
            images,
            size=(_IMAGE_SIZE, _IMAGE_SIZE),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).clamp_(0.0, 1.0)
        mean = torch.tensor(_IMAGE_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGE_STD).view(1, 3, 1, 1)
        images = (images - mean) / std
        for offset, image in enumerate(images):
            self._cache[start + offset] = image.clone()
            self._cache.move_to_end(start + offset)
        while len(self._cache) > _FRAME_CACHE:
            self._cache.popitem(last=False)

    def __getitem__(self, index: int):
        if index in self._cache:
            self._cache.move_to_end(index)
        else:
            if self._last is not None and index == self._last + 1:
                self._decode(index, min(self.count, index + _DECODE_CHUNK))
            elif self._last is not None and index == self._last - 1:
                self._decode(max(0, index - _DECODE_CHUNK + 1), index + 1)
            else:
                self._decode(index, index + 1)
        self._last = index
        return self._cache[index]

    def picture(self, index: int) -> np.ndarray:
        """Decoded frame ``index`` as 8-bit BGR, for showing to the user.

        Reads the same decoded frames ``__getitem__`` serves, without touching
        its cache or its direction of travel, so a track is not disturbed.
        """
        import torch

        frame = self._readRange(index, index + 1)[0]
        image = frame.numpy()
        if frame.dtype == torch.uint16:
            image = (image >> 8).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def pickDevice():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _copyDicts(value, levels: int):
    if isinstance(value, list):
        return list(value)
    if levels == 0 or not isinstance(value, dict):
        return value
    return {key: _copyDicts(item, levels - 1) for key, item in value.items()}


def _videoIdentity(video: str) -> dict:
    """``session.json``'s size and mtime of ``video``; empty when it cannot be read."""
    try:
        info = os.stat(video)
    except OSError:
        return {}
    return {"videoSize": info.st_size, "videoMtime": info.st_mtime}


class SamSession:
    def __init__(self, emit, cancelEvent: threading.Event):
        self.emit = emit
        self.cancelEvent = cancelEvent
        self.predictor = None
        self.modelName = None
        self.device = None
        self.frames = None
        self.state = None
        self.video = ""
        self.videoIdentity: dict = {}
        self.prompts: dict[int, dict[int, list]] = {}
        # Every state-changing request since ``open``, as executed. Sent again in
        # order to a fresh session, with each track propagate-only, it rebuilds this
        # session's tracking state.
        self.ops: list[dict] = []
        self.lastTrackStart: dict | None = None
        # (object, frame) -> the point lists sent there, each extending the one
        # before, and that frame's outputs from before the first of them.
        self.clickChains: dict[tuple[int, int], dict] = {}
        # object -> frame -> (shape, compressed bits): what a preview shows for an
        # object whose tracked output on that frame was pruned. Never read by SAM.
        self.prunedMasks: dict[int, dict] = {}
        self._prunedThisTrack: dict[int, dict] = {}
        # Scene cut detection: per-frame thumbnails and the answers already given.
        self._signatures = OrderedDict()
        self._differences: dict = {}
        # The last roto document a preview used, with its frame cache.
        self._roto = None

    # Autocast is entered per call: a generator-based track and the prompts all run
    # on this thread, and leaving it entered for good would leak into anything else.
    def _autocast(self):
        import torch

        device = self.device.type
        if device == "cuda":
            return torch.autocast(device, dtype=torch.bfloat16)
        return torch.autocast(device, enabled=False)

    def _status(self, message: str) -> None:
        self.emit({"event": "status", "message": message})

    def open(self, request: dict) -> dict:
        import torch

        from src.sam.build import MODELS, buildPredictor, weightPath

        model = request.get("model") or "anime"
        if model not in MODELS:
            raise ValueError(f"Unknown SAM model: {model}")
        video = request["video"]
        if not os.path.isfile(video):
            raise ValueError(f"Video not found: {video}")

        if self.predictor is None or self.modelName != model:
            self.predictor = None
            self.device = pickDevice()
            self._status("Getting the SAM 2 model ready...")
            checkpoint = weightPath(model)
            self._status("Loading the SAM 2 model...")
            self.predictor = buildPredictor(checkpoint, self.device)
            self.modelName = model

        self._status("Reading the video...")
        # Only replaced once it is known good, so a failed open leaves the session
        # on the video it already had.
        frames = FrameSource(video)
        if len(frames) <= 0:
            raise ValueError("The video has no frames.")
        try:
            with torch.inference_mode(), self._autocast():
                state = self.predictor.init_state(
                    frames,
                    frames.height,
                    frames.width,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=self.device.type != "cuda",
                )
        except Exception as error:
            self._releaseMemory(error)
            raise
        self.state = state
        self.frames = frames
        self.video = video
        self.videoIdentity = _videoIdentity(video)
        self.prompts = {}
        self.ops = []
        self.lastTrackStart = None
        self.clickChains = {}
        self.prunedMasks = {}
        self._signatures = OrderedDict()
        self._differences = {}
        self._roto = None
        return {
            "frames": len(self.frames),
            "width": self.frames.width,
            "height": self.frames.height,
            "fps": self.frames.fps,
            "device": self.device.type,
        }

    def _require(self) -> None:
        if self.state is None:
            raise ValueError("No video is open.")

    def prompt(self, request: dict) -> dict:
        return self._atomically(self._prompt, request)

    def _atomically(self, step, request: dict) -> dict:
        """Run a click or a removal so that a failure leaves nothing behind.

        Out of GPU memory is the realistic failure, partway through the predictor,
        and the retry has to give what the click would have given.
        """
        if self.state is None:
            return step(request)
        saved = {
            "state": self._copyState(self.state, _REQUEST_STATE),
            "prompts": _copyDicts(self.prompts, 2),
            "clickChains": {
                key: {**chain, "lists": list(chain["lists"])}
                for key, chain in self.clickChains.items()
            },
            "ops": len(self.ops),
            "prunedMasks": dict(self.prunedMasks),
        }
        try:
            return step(request)
        except Exception as error:
            self.state.update(saved["state"])
            self.prompts = saved["prompts"]
            self.clickChains = saved["clickChains"]
            self.prunedMasks = saved["prunedMasks"]
            del self.ops[saved["ops"] :]
            del saved
            self._releaseMemory(error)
            raise

    def _releaseMemory(self, error: Exception) -> None:
        """Free what a failed call allocated: its traceback still holds the tensors."""
        import torch

        traceback.clear_frames(error.__traceback__)
        gc.collect()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _prompt(self, request: dict) -> dict:
        import torch

        self._require()
        frame = int(request["frame"])
        objectId = int(request["object"])
        if not 0 <= frame < len(self.frames):
            raise ValueError(f"Frame {frame} is outside the video.")
        points = request.get("points") or []

        with torch.inference_mode(), self._autocast():
            if not points:
                self.prompts.get(objectId, {}).pop(frame, None)
                if objectId not in self.state["obj_id_to_idx"]:
                    return {"empty": True}
                self.predictor.clear_all_prompts_in_frame(
                    self.state, frame, objectId, need_output=False
                )
                chain = self.clickChains.pop((objectId, frame), None)
                if chain is not None:
                    self._restoreFrame(objectId, frame, chain["baseline"])
                # An object left with no inputs anywhere makes tracking refuse.
                if not self.prompts.get(objectId):
                    self.prompts.pop(objectId, None)
                    self.predictor.remove_object(
                        self.state, objectId, strict=False, need_output=False
                    )
                    self._forgetObject(objectId)
                # Read last, so the removed click and any removed object are gone.
                combined = self._frameMask(frame)
            else:
                masks = self._applyClicks(objectId, frame, points)
                self.prompts.setdefault(objectId, {})[frame] = points
                combined = self._withPrunedObjects(
                    frame, (masks > 0.0).any(dim=0)[0].cpu().numpy()
                )
        self.ops.append(
            {"op": "prompt", "frame": frame, "object": objectId, "points": points}
        )
        return {"mask": previewPng(combined), "area": int(combined.sum())}

    def _frameMask(self, frame: int) -> np.ndarray:
        """Every object's current mask on ``frame``, as a preview shows them."""
        return self._frameLabels(frame) > 0

    def _frameLabels(self, frame: int) -> np.ndarray:
        """Which object each pixel of ``frame`` belongs to (0 for none), as a preview
        shows them. Where objects overlap, the most likely one wins."""
        if not self.state["obj_id_to_idx"]:
            return np.zeros((self.frames.height, self.frames.width), dtype=np.uint8)
        isCond = any(
            frame in outputs["cond_frame_outputs"]
            for outputs in self.state["temp_output_dict_per_obj"].values()
        )
        consolidated = self.predictor._consolidate_temp_output_across_obj(
            self.state, frame, is_cond=isCond, consolidate_at_video_res=True
        )
        _, masks = self.predictor._get_orig_video_res_output(
            self.state, consolidated["pred_masks_video_res"]
        )
        best, index = masks.float().max(dim=0)
        idToIndex = self.state["obj_id_to_idx"]
        indexToId = self.state.get("obj_idx_to_id") or {
            i: o for o, i in idToIndex.items()
        }
        table = np.array(
            [_labelOf(indexToId.get(i, i + 1)) for i in range(masks.shape[0])],
            dtype=np.uint8,
        )
        labels = np.where(
            best[0].cpu().numpy() > 0, table[index[0].cpu().numpy()], 0
        ).astype(np.uint8)
        return self._withPrunedObjects(frame, labels)

    def _ensureConditioning(self, objectId: int, frame: int) -> None:
        """Give a click on a tracked frame the conditioning frame it has to read.

        It builds on the object's confirmed clicks. Removing or undoing the only
        confirmed one can leave clicks that are still unconfirmed, which are then
        confirmed now, or none at all, which makes this a first click on the frame.
        """
        stores = self._frameStores(objectId)
        if stores is None or frame not in stores["tracked"] or stores["cond"]:
            return
        if stores["tempCond"]:
            self.predictor.propagate_in_video_preflight(self.state)
        else:
            del stores["tracked"][frame]

    def _addPoints(self, objectId: int, frame: int, points: list):
        self._ensureConditioning(objectId, frame)
        _, _, masks = self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=frame,
            obj_id=objectId,
            points=np.array([[p["x"], p["y"]] for p in points], dtype=np.float32),
            labels=np.array(
                [1 if p.get("positive", True) else 0 for p in points],
                dtype=np.int32,
            ),
            clear_old_points=True,
        )
        return masks

    def _applyClicks(self, objectId: int, frame: int, points: list):
        """Send a frame's clicks, taking back the effect of any that were removed.

        SAM feeds a frame's last mask in with new clicks, so a list that merely
        drops a click would still carry that click's mask. A list that does not
        extend the last one therefore puts the frame back as it was before its
        first click (nothing, or the mask the track gave it) and sends the clicks
        again in their original steps. If the list was never sent on the way
        here, as when a middle dot goes, it is sent in one step.
        """
        chain = self.clickChains.get((objectId, frame))
        if chain is None:
            chain = {"lists": [], "baseline": self._frameOutputs(objectId, frame)}
            self.clickChains[(objectId, frame)] = chain
        lists = chain["lists"]
        if lists and points[: len(lists[-1])] != lists[-1]:
            lists[:] = lists[: lists.index(points)] if points in lists else []
            self._restoreFrame(objectId, frame, chain["baseline"])
            for earlier in lists:
                self._addPoints(objectId, frame, earlier)
        lists.append(points)
        return self._addPoints(objectId, frame, points)

    def _frameStores(self, objectId: int) -> dict | None:
        objectIndex = self.state["obj_id_to_idx"].get(objectId)
        if objectIndex is None:
            return None
        outputs = self.state["output_dict_per_obj"][objectIndex]
        temporary = self.state["temp_output_dict_per_obj"][objectIndex]
        return {
            "cond": outputs["cond_frame_outputs"],
            "nonCond": outputs["non_cond_frame_outputs"],
            "tempCond": temporary["cond_frame_outputs"],
            "tempNonCond": temporary["non_cond_frame_outputs"],
            "tracked": self.state["frames_tracked_per_obj"][objectIndex],
        }

    def _frameOutputs(self, objectId: int, frame: int) -> dict:
        stores = self._frameStores(objectId)
        if stores is None:
            return {}
        return {name: store[frame] for name, store in stores.items() if frame in store}

    def _restoreFrame(self, objectId: int, frame: int, saved: dict) -> None:
        stores = self._frameStores(objectId)
        if stores is None:
            return
        for name, store in stores.items():
            if name in saved:
                store[frame] = saved[name]
            else:
                store.pop(frame, None)

    def _forgetObject(self, objectId: int) -> None:
        self.prunedMasks.pop(objectId, None)
        for key in [key for key in self.clickChains if key[0] == objectId]:
            del self.clickChains[key]

    def _withPrunedObjects(self, frame: int, combined: np.ndarray) -> np.ndarray:
        """A preview's mask (or label map) plus the objects SAM no longer has an output
        for there, under their own label.

        The predictor leaves those out of what it returns, which made them vanish
        from the preview of a correction far behind the end of the track.
        """
        for objectId, byFrame in self.prunedMasks.items():
            stores = self._frameStores(objectId)
            if frame not in byFrame or stores is None:
                continue
            if any(frame in stores[name] for name in _OUTPUT_STORES):
                continue
            (height, width), data = byFrame[frame]
            bits = np.unpackbits(
                np.frombuffer(zlib.decompress(data), dtype=np.uint8),
                count=height * width,
            ).reshape(height, width)
            size = (combined.shape[1], combined.shape[0])
            resized = cv2.resize(bits * 255, size, interpolation=cv2.INTER_LINEAR) > 127
            if combined.dtype == bool:
                combined = combined | resized
            else:
                combined = np.where(
                    resized & (combined == 0), _labelOf(objectId), combined
                ).astype(np.uint8)
        return combined

    def frame(self, request: dict) -> dict:
        self._require()
        frame = int(request["frame"])
        if not 0 <= frame < len(self.frames):
            raise ValueError(f"Frame {frame} is outside the video.")
        image = self.frames.picture(frame)
        encoded, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded:
            raise RuntimeError(f"Could not encode frame {frame}.")
        return {
            "jpeg": base64.b64encode(data.tobytes()).decode("ascii"),
            "width": image.shape[1],
            "height": image.shape[0],
        }

    def remove(self, request: dict) -> dict:
        return self._atomically(self._remove, request)

    def _remove(self, request: dict) -> dict:
        import torch

        self._require()
        objectId = int(request["object"])
        known = objectId in self.prompts or objectId in self.state["obj_id_to_idx"]
        self.prompts.pop(objectId, None)
        with torch.inference_mode(), self._autocast():
            if objectId in self.state["obj_id_to_idx"]:
                self.predictor.remove_object(
                    self.state, objectId, strict=False, need_output=False
                )
            self._forgetObject(objectId)
            if not known:
                return {}
            # Every clicked frame, not only the removed object's: it may also be
            # part of another frame's preview through its tracked output.
            clicked = sorted({f for frames in self.prompts.values() for f in frames})
            previews = [
                {"frame": frame, "mask": previewPng(self._frameMask(frame))}
                for frame in clicked
            ]
        self.ops.append({"op": "remove", "object": objectId})
        return {"previews": previews}

    @staticmethod
    def _copyState(source: dict, shape: dict) -> dict:
        return {
            key: _copyDicts(source[key], levels)
            for key, levels in shape.items()
            if key in source
        }

    def _trackingSnapshot(self, source: dict) -> dict:
        return self._copyState(source, _TRACKING_STATE)

    def _prune(self, frame: int, step: int) -> None:
        """Drop tracked outputs too far behind to be read again, so memory stays flat."""
        stale = frame - step * _KEEP_TRACKED_FRAMES
        for objectIndex, outputs in self.state["output_dict_per_obj"].items():
            pruned = outputs["non_cond_frame_outputs"].pop(stale, None)
            if isinstance(pruned, dict):
                self._rememberPruned(objectIndex, stale, pruned["pred_masks"])

    def _rememberPruned(self, objectIndex: int, frame: int, masks) -> None:
        objectId = self.state["obj_idx_to_id"][objectIndex]
        shape = tuple(masks.shape[-2:])
        data = zlib.compress(np.packbits((masks[0, 0] > 0).cpu().numpy()).tobytes(), 1)
        self._prunedThisTrack.setdefault(objectId, {})[frame] = (shape, data)

    def _signature(self, frame: int):
        """A tiny thumbnail of ``frame`` for spotting scene cuts, from the frames the
        predictor already decoded."""
        import torch.nn.functional as F

        cached = self._signatures.get(frame)
        if cached is None:
            image = self.frames[frame].float().unsqueeze(0)
            cached = F.adaptive_avg_pool2d(image, _CUT_GRID)[0]
            self._signatures[frame] = cached
            # Only the differences are kept for good; a thumbnail is needed again only
            # for the frames next to it.
            while len(self._signatures) > _CUT_SIGNATURES:
                self._signatures.popitem(last=False)
        else:
            self._signatures.move_to_end(frame)
        return cached

    def _difference(self, a: int) -> float:
        """How much frame ``a + 1`` differs from frame ``a``; 0 off either end."""
        if a < 0 or a + 1 >= len(self.frames):
            return 0.0
        known = self._differences.get(a)
        if known is None:
            known = (self._signature(a) - self._signature(a + 1)).abs().mean().item()
            self._differences[a] = known
        return known

    def _isCut(self, a: int, b: int) -> bool:
        """Whether a scene cut separates neighbouring frames ``a`` and ``b``.

        A cut is one change far bigger than the typical change around it: the median
        of the changes within ``_CUT_REACH`` frames, leaving out held frames (anime
        on twos or threes repeats each drawing, so most neighbours change by
        nothing). A fixed threshold cannot tell a cut from fast action, and the
        largest neighbour cannot either: a drawing change right after a cut, or a
        moving shot before it, would hide the cut.
        """
        low = min(a, b)
        change = self._difference(low)
        if change <= _CUT_THRESHOLD:
            return False
        near = [
            d
            for other in range(low - _CUT_REACH, low + _CUT_REACH + 1)
            if other != low and (d := self._difference(other)) > _CUT_FLOOR
        ]
        around = statistics.median(near) if near else _CUT_FLOOR
        return change > _CUT_RATIO * max(around, _CUT_FLOOR)

    def _pass(self, job: dict, origin: int, reverse: bool, limit: int) -> int | None:
        """Propagate from ``origin`` toward ``limit``. Returns the first frame of the
        next shot when a scene cut stopped it, else None."""
        count = origin - limit if reverse else limit - origin
        if count < 0:
            return None
        if reverse and origin == 0:
            # The predictor skips a reverse pass that starts at frame 0, and frame 0 is
            # all such a pass covers.
            reverse, count = False, 0
        previous = None
        for frame, ids, masks in self.predictor.propagate_in_video(
            self.state,
            start_frame_idx=origin,
            max_frame_num_to_track=count,
            reverse=reverse,
        ):
            if self.cancelEvent.is_set():
                raise Cancelled()
            if (
                job["stopAtCuts"]
                and previous is not None
                and self._isCut(previous, frame)
            ):
                job["cuts"].add(max(previous, frame))
                # The predictor has already tracked this frame from the other shot. Left
                # in, a click there would start from that mask, and the next pass would
                # read it as memory.
                self._forgetTracked(frame)
                return frame
            previous = frame
            if job["start"] <= frame <= job["end"]:
                if job["output"] is not None:
                    self._writeMask(job["output"], frame, ids, masks)
                if frame not in job["written"]:
                    job["written"].add(frame)
                    job["report"]()
            self._prune(frame, -1 if reverse else 1)
        return None

    def _runPasses(self, job: dict, direction: str, origin: int, anchors: list) -> None:
        start, end = job["start"], job["end"]
        if direction == "forward":
            self._pass(job, origin, False, end)
            return
        if direction == "backward":
            self._pass(job, origin, True, start)
            return
        first = anchors[0]
        if first > start:
            self._pass(job, first, True, start)
        at = first
        while True:
            stop = self._pass(job, at, False, end)
            if stop is None:
                return
            # The next shot is tracked only from a click inside it.
            following = [a for a in anchors if a >= stop]
            if not following or following[0] > end:
                return
            nxt = following[0]
            if nxt > stop:
                self._pass(job, nxt, True, stop)
            at = nxt

    def _forgetTracked(self, frame: int) -> None:
        """Drop what tracking stored for ``frame``, except a clicked frame's own output."""
        for objectIndex, outputs in self.state["output_dict_per_obj"].items():
            outputs["non_cond_frame_outputs"].pop(frame, None)
            temporary = self.state["temp_output_dict_per_obj"].get(objectIndex)
            if temporary is not None:
                temporary["non_cond_frame_outputs"].pop(frame, None)
            if frame not in outputs["cond_frame_outputs"]:
                self.state["frames_tracked_per_obj"].get(objectIndex, {}).pop(
                    frame, None
                )

    @staticmethod
    def _writeMask(output: str, frame: int, ids, masks) -> None:
        # Every object in one matte: the most likely one decides each pixel.
        best, index = masks.float().max(dim=0)
        _writePng(
            os.path.join(output, maskFileName(frame)),
            matteImage(edgeCoverage(best[0]).cpu().numpy()),
        )
        # Which object each pixel belongs to, for the front end's per-object colours.
        # TAS's render never reads it.
        table = np.array([_labelOf(i) for i in ids], dtype=np.uint8)
        labels = np.where(
            best[0].cpu().numpy() > 0, table[index[0].cpu().numpy()], 0
        ).astype(np.uint8)
        _writePng(os.path.join(output, LABELS_DIR, maskFileName(frame)), labels)

    @staticmethod
    def _carryOver(base: str, output: str) -> dict:
        """Hard links (or copies) of an earlier track's mattes, and its session file."""
        session = {}
        try:
            with open(os.path.join(base, SESSION_FILE), encoding="utf-8") as handle:
                session = json.load(handle)
        except OSError, ValueError:
            session = {}
        for folder in ("", LABELS_DIR):
            source = os.path.join(base, folder)
            if not os.path.isdir(source):
                continue
            os.makedirs(os.path.join(output, folder), exist_ok=True)
            for frame, path in listMaskFrames(source).items():
                target = os.path.join(output, folder, maskFileName(frame))
                if os.path.exists(target):
                    continue
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)
        return session if isinstance(session, dict) else {}

    def track(self, request: dict) -> dict:
        # Not cleared on the way in: a cancel can arrive while this request is
        # still waiting for the one before it, and that cancel is for this track.
        try:
            return self._track(request)
        except Exception as error:
            self._releaseMemory(error)
            raise
        finally:
            self.cancelEvent.clear()

    def _track(self, request: dict) -> dict:
        import torch

        self._require()
        start = max(0, int(request["start"]))
        end = min(len(self.frames) - 1, int(request["end"]))
        if start > end:
            raise ValueError(f"Nothing to track from frame {start} to frame {end}.")
        output = request["output"]
        direction = request.get("direction") or "both"
        if direction not in ("both", "forward", "backward"):
            raise ValueError(f"Unknown tracking direction: {direction}")
        stopAtCuts = request.get("stopAtCuts") is True
        base = request.get("base") if direction != "both" else None
        promptFrames = sorted({f for frames in self.prompts.values() for f in frames})
        if not promptFrames:
            raise ValueError("Click on an object first.")
        first = promptFrames[0]
        origin = first
        if direction == "both":
            if first > end:
                raise ValueError("Every clicked frame is after the range to track.")
        else:
            origin = min(end, max(start, int(request.get("from", start))))
        if output is not None:
            os.makedirs(os.path.join(output, LABELS_DIR), exist_ok=True)
        if self.cancelEvent.is_set():
            return {"cancelled": True}

        total = (
            end - origin + 1
            if direction == "forward"
            else origin - start + 1
            if direction == "backward"
            else end - start + 1
        )
        done = [0]

        def report():
            done[0] += 1
            self.emit({"event": "progress", "done": done[0], "total": total})

        objects = {
            str(objectId): {str(f): pts for f, pts in frames.items()}
            for objectId, frames in self.prompts.items()
        }
        # A track that does not complete is not logged, so nothing replays it. It
        # must not leave anything behind that later clicks would build on either.
        before = self._trackingSnapshot(self.state)
        # Tracking again on top of its own results drifts: a clicked frame after the
        # range start reads what the previous reverse pass left. The same round
        # therefore runs from the same state, and stays one entry in the log.
        entry = {"op": "track", "start": start, "end": end}
        if direction != "both":
            entry["direction"] = direction
            entry["from"] = origin
        if stopAtCuts:
            entry["stopAtCuts"] = True
        repeated = bool(self.ops) and self.ops[-1] == entry
        self._prunedThisTrack = {}
        if repeated:
            self.state.update(self._trackingSnapshot(self.lastTrackStart))
        carried = {}
        if output is not None and base:
            carried = self._carryOver(base, output)
        job = {
            "start": start,
            "end": end,
            "output": output,
            "stopAtCuts": stopAtCuts,
            "cuts": set(),
            "written": set(),
            "report": report,
        }
        completed = False
        try:
            try:
                with torch.inference_mode(), self._autocast():
                    self._runPasses(job, direction, origin, promptFrames)
            except Cancelled:
                return {"cancelled": True}
            ops = self.ops if repeated else [*self.ops, entry]
            spans = _mergeSpans(
                _spansOf(job["written"]) + _validSpans(carried.get("spans"))
            )
            # A cut the earlier track stopped at stays only if this track did not run
            # straight through it (writing the frames on both sides).
            written = job["written"]
            kept = {
                c
                for c in carried.get("cuts", [])
                if isinstance(c, int) and not (c in written and c - 1 in written)
            }
            cuts = sorted(job["cuts"] | kept)
            if carried:
                start = min(start, int(carried.get("start", start)))
                end = max(end, int(carried.get("end", end)))
            if output is not None:
                self._writeSessionFile(output, start, end, objects, ops, spans, cuts)
            self.ops = ops
            for objectId, byFrame in self._prunedThisTrack.items():
                self.prunedMasks.setdefault(objectId, {}).update(byFrame)
            if not repeated:
                self.lastTrackStart = before
            completed = True
        finally:
            if not completed:
                self.state.update(before)
        return {
            "start": start,
            "end": end,
            "output": output,
            "spans": spans,
            "cuts": cuts,
        }

    def _writeSessionFile(self, output, start, end, objects, ops, spans, cuts) -> None:
        session = {
            "version": 1,
            "video": self.video,
            **self.videoIdentity,
            "model": self.modelName,
            "width": self.frames.width,
            "height": self.frames.height,
            "frameCount": len(self.frames),
            "start": start,
            "end": end,
            "objects": objects,
            "ops": ops,
            "spans": spans,
            "cuts": cuts,
        }
        temporary = os.path.join(output, SESSION_FILE + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(session, handle)
        os.replace(temporary, os.path.join(output, SESSION_FILE))

    def composite(self, request: dict) -> dict:
        """The final matte of a frame for a roto document: mattes, shapes and Refine
        Edge, the render's own code. ``live`` takes the frame's matte from the clicks
        not tracked yet instead of from the folder."""
        from src.segment.rotoDoc import RotoRenderer, parseRotoDoc

        self._require()
        frame = int(request["frame"])
        if not 0 <= frame < len(self.frames):
            raise ValueError(f"Frame {frame} is outside the video.")
        raw = request.get("doc")
        # A front end names each version of its document and sends the document only
        # when that name changes, which spares sending and parsing a tracked shape's
        # thousands of keys on every frame change. Without a name the document itself
        # is the key.
        docKey = request.get("docKey")
        named = isinstance(docKey, str) and docKey != ""
        key = f"key:{docKey}" if named else json.dumps(raw, sort_keys=True)
        if self._roto is None or self._roto[0] != key:
            if raw is None:
                raise ValueError(ROTO_DOC_MISSING)
            self._roto = (key, RotoRenderer(parseRotoDoc(raw)))
        renderer = self._roto[1]
        height, width = self.frames.height, self.frames.width
        base = None
        labels = None
        if request.get("live"):
            import torch

            with torch.inference_mode(), self._autocast():
                labels = self._frameLabels(frame)
            base = (labels > 0).astype(np.float32)
        elif request.get("labels") and renderer.doc.get("mattes"):
            from src.masking import readMaskImage

            path = os.path.join(renderer.doc["mattes"], LABELS_DIR, maskFileName(frame))
            labels = readMaskImage(path) if os.path.isfile(path) else None
        alpha = renderer.alpha(frame, height, width, base)
        scale = min(1.0, _PREVIEW_MAX_SIDE / max(height, width))
        size = (round(width * scale), round(height * scale))
        image = np.rint(alpha * 255.0).astype(np.uint8)
        if scale < 1.0:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        reply = {"matte": _pngBase64(image)}
        if request.get("labels") and labels is not None and labels.ndim == 2:
            if labels.shape[:2] != (size[1], size[0]):
                labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
            reply["labels"] = _pngBase64(labels.astype(np.uint8))
        return reply

    def trackShape(self, request: dict) -> dict:
        """Follow points from frame ``frame`` to frame ``to`` with optical flow, for a
        hand-drawn shape that should move with the picture. Answers with the points on
        every frame on the way, ``to`` included; with ``stopAtCuts``, it stops before a
        scene cut. Leaves the tracking state alone."""
        try:
            return self._trackShape(request)
        finally:
            self.cancelEvent.clear()

    def _trackShape(self, request: dict) -> dict:
        self._require()
        start = int(request["frame"])
        target = int(request["to"])
        count = len(self.frames)
        if not (0 <= start < count and 0 <= target < count) or start == target:
            raise ValueError("Pick another frame to track the shape to.")
        raw = request.get("points") or []
        if not isinstance(raw, list) or not raw or len(raw) > _MAX_SHAPE_POINTS:
            raise ValueError("The shape has no points to track.")
        points = np.array([[float(p[0]), float(p[1])] for p in raw], dtype=np.float32)
        step = 1 if target > start else -1
        stopAtCuts = request.get("stopAtCuts") is True
        total = abs(target - start)
        width, height = self.frames.width, self.frames.height
        previous = cv2.cvtColor(self.frames.picture(start), cv2.COLOR_BGR2GRAY)
        tracked = []
        stoppedAt = None
        for n, frame in enumerate(range(start + step, target + step, step), start=1):
            if self.cancelEvent.is_set():
                return {"cancelled": True}
            if stopAtCuts and self._isCut(frame - step, frame):
                stoppedAt = frame
                break
            current = cv2.cvtColor(self.frames.picture(frame), cv2.COLOR_BGR2GRAY)
            points = _flowPoints(previous, current, points, width, height)
            tracked.append({"frame": frame, "points": np.round(points, 2).tolist()})
            previous = current
            self.emit({"event": "progress", "done": n, "total": total})
        return {"frames": tracked, "stoppedAt": stoppedAt}


def _labelOf(objectId) -> int:
    """An object id as a label byte: 0 means no object, so ids map to 1..255."""
    return min(255, max(1, int(objectId)))


def _writePng(path: str, image: np.ndarray) -> None:
    """``image`` to ``path``, through a temporary name, so a frame carried over from an
    earlier track as a hard link is replaced, never written through. imencode plus a
    Python write, because cv2.imwrite cannot open a non-ASCII Windows path and the
    folder is under the user's home."""
    encoded, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not encoded:
        raise OSError(f"Could not encode {path}")
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(data.tobytes())
    os.replace(temporary, path)


def _pngBase64(image: np.ndarray) -> str:
    encoded, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not encoded:
        raise RuntimeError("Could not encode an image.")
    return base64.b64encode(data.tobytes()).decode("ascii")


def _flowPoints(previous, current, points, width, height):
    """``points`` moved from frame ``previous`` to ``current`` (grey images) with
    pyramidal Lucas-Kanade flow. A point that cannot be followed, or lies off the
    picture, moves with the median of the points that could."""
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    moved = points.copy()
    good = np.zeros(len(points), dtype=bool)
    if inside.any():
        start = points[inside].reshape(-1, 1, 2)
        params = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, start, None, **params
        )
        back, backStatus, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, forward, None, **params
        )
        # Forward then back should land where it started; drift means a lost point.
        agree = np.linalg.norm(back.reshape(-1, 2) - start.reshape(-1, 2), axis=1) < 1.5
        ok = (status.reshape(-1) == 1) & (backStatus.reshape(-1) == 1) & agree
        indices = np.flatnonzero(inside)
        moved[indices[ok]] = forward.reshape(-1, 2)[ok]
        good[indices[ok]] = True
    if good.any():
        shift = np.median(moved[good] - points[good], axis=0)
        moved[~good] = points[~good] + shift
    return moved


def _spansOf(frames) -> list:
    spans = []
    for frame in sorted(frames):
        if spans and frame == spans[-1][1] + 1:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return spans


def _validSpans(raw) -> list:
    if not isinstance(raw, list):
        return []
    return [
        [int(a), int(b)]
        for a, b in (s for s in raw if isinstance(s, list) and len(s) == 2)
        if isinstance(a, int) and isinstance(b, int) and 0 <= a <= b
    ]


def _mergeSpans(spans: list) -> list:
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


class TrackCancel:
    """Decides which ``cancel`` lines stop a track.

    A cancel is read on another thread, so it can land before the track it was
    sent for has started, or after that track has already finished. It counts
    while a track has been requested and not yet answered; otherwise there is
    nothing to stop, and setting the event would abort whichever track comes next.
    """

    def __init__(self, event: threading.Event):
        self.event = event
        self._lock = threading.Lock()
        self._outstanding = 0

    def trackRequested(self) -> None:
        with self._lock:
            self._outstanding += 1

    def cancel(self) -> None:
        with self._lock:
            if self._outstanding:
                self.event.set()

    def trackAnswered(self) -> None:
        # ``track`` clears the event on its way out, but a cancel can still land
        # between that and the reply, so the last clear has to be under the lock.
        with self._lock:
            self._outstanding -= 1
            if not self._outstanding:
                self.event.clear()


def runSession() -> int:
    """Serve requests until stdin closes or ``quit``. Returns the exit code."""
    out = sys.stdout
    # Anything else that prints (a download bar, a library warning) must not land
    # between protocol lines.
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(message: dict) -> None:
        with lock:
            out.write(PREFIX + json.dumps(message) + "\n")
            out.flush()

    requests: queue.Queue = queue.Queue()
    cancelEvent = threading.Event()
    trackCancel = TrackCancel(cancelEvent)

    def readRequests() -> None:
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                except ValueError:
                    request = None
                if not isinstance(request, dict):
                    emit({"event": "status", "message": "Ignored a malformed request."})
                    continue
                if request.get("cmd") == "cancel":
                    trackCancel.cancel()
                    continue
                if request.get("cmd") in _CANCELLABLE:
                    trackCancel.trackRequested()
                requests.put(request)
        finally:
            cancelEvent.set()
            requests.put(None)

    threading.Thread(target=readRequests, daemon=True).start()

    session = SamSession(emit, cancelEvent)
    handlers = {
        "open": session.open,
        "prompt": session.prompt,
        "frame": session.frame,
        "remove": session.remove,
        "track": session.track,
        "composite": session.composite,
        "trackShape": session.trackShape,
    }
    emit({"event": "ready"})
    while True:
        request = requests.get()
        if request is None or request.get("cmd") == "quit":
            return 0
        requestId = request.get("id")
        try:
            # Inside the try: a ``cmd`` that is a list or object is unhashable.
            handler = handlers.get(request.get("cmd"))
            if handler is None:
                raise ValueError(f"Unknown command: {request.get('cmd')}")
            emit({"id": requestId, "ok": True, **handler(request)})
        except Exception as error:
            logging.exception("SAM session request failed")
            message = str(error) or type(error).__name__
            emit({"id": requestId, "ok": False, "error": message})
        finally:
            if request.get("cmd") in _CANCELLABLE:
                trackCancel.trackAnswered()
