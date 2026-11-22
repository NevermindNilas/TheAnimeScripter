"""SMoVSR (``smovsr*``): the bidirectional 2x anime video upscaler, every backend.

The network reads a clip forwards and then backwards (see
``src/extraArches/smovsr.py``), so one output frame depends on frames after it.
Every backend here streams it the same way, in windows of ``WINDOW_FRAMES``
source frames of which the first ``OUTPUT_FRAMES`` are written:

- the stage declares ``temporalWindow = (0, WINDOW_FRAMES - 1)`` and
  ``takesLookahead``, so the pipeline hands each centre the frames after it in
  its shot (``Window.ahead``);
- on the first centre of a window the driver runs the window and queues its
  outputs; the next ``OUTPUT_FRAMES - 1`` centres are served from the queue;
- the forward recurrence's state is carried into the next window; a scene cut
  or the stream end inside a window closes that window's gates, so the shot
  before it is finished as if the clip ended there and the next one starts
  fresh. A short window is padded with its last frame behind closed gates.

Measured on 100 frames of anime at 480x270 with two cuts, fp32, against the
upstream forward over each whole shot: mean |diff| 0.13/255, worst frame
0.29/255, and the largest frame-to-frame change the windows add is 0.23/255
(non-overlapping 30-frame clips without a carried state: 1.08/255). A shot
shorter than one window matches the upstream forward to 1e-4/255.
"""

import logging
from collections import deque

import torch

from src.constants import ADOBE
from src.infra.logAndPrint import logAndPrint
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

WINDOW_FRAMES = 12
OUTPUT_FRAMES = 8
STATE_CHANNELS = 32
SCALE = 2


class SMoVSRStream:
    """The window schedule shared by every backend; subclasses run one window.

    ``_runWindow(clip, gates, count)`` takes the padded clip ``(1, T, 3, H, W)``
    and the gate row ``(1, T + 1)`` as host lists/tensors prepared here, updates
    the backend's own forward state, and returns ``count`` output frames
    ``(1, 3, 2H, 2W)`` that nothing will overwrite later.
    """

    temporalWindow = (0, WINDOW_FRAMES - 1)
    takesLookahead = True
    upscaleFactor = SCALE

    def _initStream(self) -> None:
        self._queue: deque = deque()
        self._continues = False

    def __call__(self, frame, ahead=()):
        if not self._queue:
            self._startWindow(frame, ahead or ())
        return self._queue.popleft()

    def _startWindow(self, frame, ahead) -> None:
        frames = (frame, *tuple(ahead)[: WINDOW_FRAMES - 1])
        count = min(OUTPUT_FRAMES, len(frames))
        gates = windowGates(len(frames), self._continues)
        outputs = self._runWindow(frames, gates, count)
        if len(outputs) != count:
            raise RuntimeError(
                f"SMoVSR returned {len(outputs)} frames for a window of {count}"
            )
        # Frames past this window's outputs are the same shot: the next window
        # continues the forward state. Otherwise the next frame opens a new
        # shot (or there is none).
        self._continues = len(frames) > OUTPUT_FRAMES
        self._queue.extend(outputs)

    def _runWindow(self, frames, gates, count):
        raise NotImplementedError


def windowGates(validFrames: int, continues: bool) -> list[float]:
    """The ``gates`` row of one window (see ``SMoVSRWindow``).

    ``gates[0]`` continues the carried forward state; ``gates[1 + i]`` carries
    the backward recurrence from frame ``i + 1`` into frame ``i``, open only
    while frame ``i + 1`` is a real frame of the same shot.
    """
    if not 1 <= validFrames <= WINDOW_FRAMES:
        raise ValueError(f"a window holds 1..{WINDOW_FRAMES} frames, got {validFrames}")
    return [1.0 if continues else 0.0] + [
        1.0 if i + 1 < validFrames else 0.0 for i in range(WINDOW_FRAMES)
    ]


def _padded(frames, device, dtype):
    """``frames`` as one ``(1, T, 3, H, W)`` clip, the last frame repeated to fill it."""
    clip = [f.to(device=device, dtype=dtype) for f in frames]
    clip.extend([clip[-1]] * (WINDOW_FRAMES - len(clip)))
    return torch.stack([f.reshape(f.shape[-3:]) for f in clip]).unsqueeze(0)


def _pthPath() -> str:
    filename = modelsMap("smovsr", SCALE, modelType="pth")
    return resolveWeightPath("smovsr", filename, downloadModel="smovsr")


def _onnxPath(method: str, half: bool) -> str:
    registryName = method.replace("-openvino", "-directml")
    filename = modelsMap(registryName, SCALE, modelType="onnx", half=half)
    return resolveWeightPath(
        "smovsr-onnx",
        filename,
        downloadModel=registryName,
        modelType="onnx",
        half=half,
    )


class SMoVSRCuda(SMoVSRStream):
    """Eager PyTorch on CUDA, channels_last, on a private stream."""

    def __init__(self, half: bool, width: int, height: int):
        self.half = half
        self.width = width
        self.height = height
        self.handleModel()
        self._initStream()

    def _device(self):
        return torch.device("cuda")

    def handleModel(self):
        from src.extraArches.smovsr import SMoVSRWindow, loadSMoVSR

        if ADOBE:
            progressState.update({"status": "Loading SMoVSR upscale model..."})
        self.device = self._device()
        self.dtype = torch.float16 if self.half else torch.float32
        net = loadSMoVSR(_pthPath())
        self.model = SMoVSRWindow(net, WINDOW_FRAMES, OUTPUT_FRAMES)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.state = torch.zeros(
            (1, STATE_CHANNELS, self.height, self.width),
            device=self.device,
            dtype=self.dtype,
        )
        self.stream = torch.cuda.Stream()

    def _infer(self, frames, gates, count):
        clip = _padded(frames, self.device, self.dtype)
        gateRow = torch.tensor([gates], device=self.device, dtype=self.dtype)
        out, self.state = self.model(clip, self.state, gateRow)
        return [out[:, i].contiguous() for i in range(count)]

    @torch.inference_mode()
    def _runWindow(self, frames, gates, count):
        from src.infra.cudaStreams import waitForCaller

        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            outputs = self._infer(frames, gates, count)
        # The writer copies on its own stream and never waits on ours.
        self.stream.synchronize()
        return outputs


class SMoVSRROCm(SMoVSRCuda):
    """ROCm (HIP): the CUDA path eagerly on the default stream."""

    def handleModel(self):
        super().handleModel()
        self.stream = None

    @torch.inference_mode()
    def _runWindow(self, frames, gates, count):
        outputs = self._infer(frames, gates, count)
        torch.cuda.current_stream().synchronize()
        return outputs


class SMoVSRMPS(SMoVSRCuda):
    """Apple Silicon (MPS): eager, frames handed on as CPU tensors like the other MPS upscalers."""

    def _device(self):
        return torch.device("mps")

    def handleModel(self):
        super().handleModel()
        self.stream = None

    @torch.inference_mode()
    def _runWindow(self, frames, gates, count):
        outputs = self._infer(frames, gates, count)
        torch.mps.synchronize()
        return [frame.cpu() for frame in outputs]


class SMoVSRTensorRT(SMoVSRStream):
    """TensorRT: one static engine per input size over the exported window."""

    def __init__(self, half: bool, width: int, height: int):
        self.half = half
        self.width = width
        self.height = height
        self.handleModel()
        self._initStream()

    def handleModel(self):
        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        if ADOBE:
            progressState.update(
                {"status": "Loading TensorRT upscale model: smovsr-tensorrt..."}
            )
        self.dtype = torch.float16 if self.half else torch.float32
        self.modelPath = _onnxPath("smovsr-tensorrt", self.half)
        clipShape = [1, WINDOW_FRAMES, 3, self.height, self.width]
        stateShape = [1, STATE_CHANNELS, self.height, self.width]
        gateShape = [1, WINDOW_FRAMES + 1]
        enginePath = tensorRTEngineNameHandler(
            modelPath=self.modelPath, fp16=self.half, optInputShape=clipShape
        )
        self.engine, self.context = tensorRTEngineLoader(enginePath)
        if self.engine is None or self.context is None:
            shapes = [clipShape, stateShape, gateShape]
            self.engine, self.context = tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=shapes,
                inputsOpt=shapes,
                inputsMax=shapes,
                inputName=["clip", "state", "gates"],
                isMultiInput=True,
                forceStatic=True,
            )
            if self.engine is None or self.context is None:
                from src.upscale.tensorrt import _tensorRTBuildFailureMessage

                raise RuntimeError(_tensorRTBuildFailureMessage(self.modelPath))

        device = torch.device("cuda")

        def buffer(shape):
            return torch.zeros(shape, device=device, dtype=self.dtype)

        self.clip = buffer(clipShape)
        self.state = buffer(stateShape)
        self.gates = buffer(gateShape)
        self.frames = buffer(
            [1, OUTPUT_FRAMES, 3, self.height * SCALE, self.width * SCALE]
        )
        self.stateOut = buffer(stateShape)
        tensors = {
            "clip": self.clip,
            "state": self.state,
            "gates": self.gates,
            "frames": self.frames,
            "stateOut": self.stateOut,
        }
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in tensors:
                raise RuntimeError(
                    f"SMoVSR TensorRT engine has an unknown tensor {name}"
                )
            self.context.set_tensor_address(name, tensors[name].data_ptr())
            if name in ("clip", "state", "gates"):
                self.context.set_input_shape(name, tuple(tensors[name].shape))
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def _runWindow(self, frames, gates, count):
        from src.infra.cudaStreams import waitForCaller

        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            self.clip.copy_(_padded(frames, self.clip.device, self.dtype))
            self.gates.copy_(
                torch.tensor([gates], dtype=self.dtype), non_blocking=False
            )
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.state.copy_(self.stateOut)
            # The engine writes into the same buffer every window.
            outputs = [self.frames[:, i].clone() for i in range(count)]
        self.stream.synchronize()
        return outputs


class SMoVSRDirectML(SMoVSRStream):
    """DirectML / OpenVINO through ONNX Runtime; frames are handed on as CPU tensors."""

    def __init__(self, upscaleMethod: str, half: bool, width: int, height: int):
        import numpy as np
        import onnxruntime as ort

        if "openvino" in upscaleMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.np = np
        self.ort = ort
        self.ort.set_default_logger_severity(3)
        self.upscaleMethod = upscaleMethod
        self.half = half
        self.width = width
        self.height = height
        self.handleModel()
        self._initStream()

    def handleModel(self):
        from src.infra.providerCheck import warnCpuFallback, warnIfProviderMissing

        if ADOBE:
            progressState.update(
                {"status": f"Loading upscale model: {self.upscaleMethod}..."}
            )
        modelPath = _onnxPath(self.upscaleMethod, self.half)
        providers = self.ort.get_available_providers()
        if "directml" in self.upscaleMethod and "DmlExecutionProvider" in providers:
            self.model = self.ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
            warnIfProviderMissing(
                self.model, "DmlExecutionProvider", "DirectML SMoVSR upscale"
            )
        elif (
            "openvino" in self.upscaleMethod
            and "OpenVINOExecutionProvider" in providers
        ):
            self.model = self.ort.InferenceSession(
                modelPath,
                providers=[
                    ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                ],
            )
            warnIfProviderMissing(
                self.model, "OpenVINOExecutionProvider", "OpenVINO SMoVSR upscale"
            )
        else:
            warnCpuFallback(
                "DirectML/OpenVINO provider not available, falling back to CPU"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )
        self.npDType = self.np.float16 if self.half else self.np.float32
        self.torchDType = torch.float16 if self.half else torch.float32
        self.state = self.np.zeros(
            (1, STATE_CHANNELS, self.height, self.width), self.npDType
        )
        logging.info(f"SMoVSR ORT providers: {self.model.get_providers()}")

    @torch.inference_mode()
    def _runWindow(self, frames, gates, count):
        clip = _padded(frames, torch.device("cpu"), self.torchDType)
        out, stateOut = self.model.run(
            ["frames", "stateOut"],
            {
                "clip": clip.numpy(),
                "state": self.state,
                "gates": self.np.asarray([gates], self.npDType),
            },
        )
        self.state = stateOut
        frames = torch.from_numpy(out)
        return [frames[:, i].contiguous() for i in range(count)]
