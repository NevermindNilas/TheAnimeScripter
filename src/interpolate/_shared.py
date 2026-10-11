"""Shared RIFE arch table and lazy arch importer.

Deliberately free of torch/CUDA imports: the arch imports below are all
function-level lazy, and any module-level torch import here would break
torch-less loading (bare CI venv). Backend constructors own
`set_float32_matmul_precision`, not this module.
"""

_RIFE_V1 = {
    # The bare "rife" alias resolves to 4.22 in modelsMap() (it returns
    # rife422.pth / rife-v4.22-ncnn.zip), so its arch has to be 4.22 too --
    # pointing it at IFNet425 loaded rife422.pth into a net with 30 extra
    # parameters and blew up in load_state_dict (reachable via "rife-mps").
    "rife": ("IFNet422", "IFNet_rife422"),
    "rife4.25": ("IFNet425", "IFNet_rife425"),
    "rife4.25-heavy": ("IFNet425Heavy", "IFNet_rife425heavy"),
    "rife4.25-lite": ("IFNet425Lite", "IFNet_rife425lite"),
    # 4.26 is 4.25's architecture with new weights; 4.26-heavy is not 4.25-heavy.
    "rife4.26": ("IFNet426", "IFNet_rife425"),
    "rife4.26-heavy": ("IFNet426Heavy", "IFNet_rife426heavy"),
    "rife4.22": ("IFNet422", "IFNet_rife422"),
    "rife4.22-lite": ("IFNet422Lite", "IFNet_rife422lite"),
    "rife4.21": ("IFNet421", "IFNet_rife421"),
    "rife4.20": ("IFNet420", "IFNet_rife420"),
    "rife4.18": ("IFNet418", "IFNet_rife418"),
    "rife4.17": ("IFNet417", "IFNet_rife417"),
    "rife4.15-lite": ("IFNet415Lite", "IFNet_rife415lite"),
    "rife4.16-lite": ("IFNet416Lite", "IFNet_rife416lite"),
    "rife4.6": ("IFNet46", "IFNet_rife46"),
    # rife_elexor's --ensemble still runs the reference arch (see _loadV1)
    "rife_elexor": ("IFNetElexor", "IFNet_elexor_cuda"),
}


def _loadV1(method, half, staticStep=False, ensemble=False):
    # `half` no longer selects the arch: the unified fast arch is exact in fp32
    # too (and ~1.5x faster than the reference arch there). Only --static_step
    # still needs the reference arch's bakedTimestep.
    fastName, baseMod = _RIFE_V1[method]
    if fastName and not staticStep and not (ensemble and method == "rife_elexor"):
        from src.rifearches import rife_fast

        return getattr(rife_fast, fastName)
    mod = __import__(f"src.rifearches.{baseMod}", fromlist=["IFNet"])
    return mod.IFNet


# TensorRT methods run the SAME arch as the CUDA path -- rife_fast's variant
# table and blocks -- lowered for ONNX by rife_fast.exportArch. 4.15 (full)
# shares 4.17's architecture. DirectML/OpenVINO still use Rife_directml
# (_DML_LEGACY) for now.
_EXPORT_FAST = {
    "rife": "IFNet422",
    "rife4.6": "IFNet46",
    "rife4.15": "IFNet417",
    "rife4.17": "IFNet417",
    "rife4.18": "IFNet418",
    "rife4.20": "IFNet420",
    "rife4.21": "IFNet421",
    "rife4.22": "IFNet422",
    "rife4.22-lite": "IFNet422Lite",
    "rife4.25": "IFNet425",
    "rife4.25-lite": "IFNet425Lite",
    "rife4.25-heavy": "IFNet425Heavy",
    "rife4.26": "IFNet426",
    "rife4.26-heavy": "IFNet426Heavy",
    "rife_elexor": "IFNetElexor",
}
# The wired set per backend (unchanged from the per-file arches this replaced).
_EXPORT_WIRED = {
    "-tensorrt": frozenset(_EXPORT_FAST),
}
# DirectML / OpenVINO arch class per base method (src/rifearches/Rife_directml.py).
_DML_LEGACY = {
    "rife4.6": "IFNet_46",
    "rife4.15": "IFNet_415",
    "rife4.17": "IFNet_415",
    "rife4.18": "IFNet_415",
    "rife4.20": "IFNet_420",
    "rife4.21": "IFNet_420",
    "rife4.22": "IFNet_422",
    "rife4.22-lite": "IFNet_422_lite",
    "rife4.25": "IFNet_425",
    "rife4.26": "IFNet_425",
    "rife4.26-heavy": "IFNet_426_heavy",
    "rife4.25-lite": "IFNet_425_lite",
    "rife4.25-heavy": "IFNet_425_heavy",
}
# TensorRT lowering on top of rife_fast._ExportIFNet's defaults (the CUDA path's
# rewrites, low-res tail included). TensorRT keeps f0 outside the engine
# (featureIO). A TensorRT 11.3 crash with the low-res tail was suspected but not
# reproduced (4.22-lite@736: isolated build + 600 replays, no CUDA error).
_EXPORT_PROFILE = {
    "-tensorrt": {"featureIO": True},
}


def _tensorrtProfile():
    """warpPad: TensorRT 10.16 miscompiled a 7-channel GridSample -> Resize in one
    older graph shape; the current graph is valid unpadded on 10.16 and 11.3, and
    padding costs ~13.5% on 11.3. TensorRT 10 keeps the padding as a conservative
    default."""
    profile = dict(_EXPORT_PROFILE["-tensorrt"])
    try:
        from src.utils.tensorrt_import import trt

        major = int(str(trt.__version__).split(".")[0])
    except Exception:
        major = 0
    profile["warpPad"] = 1 if major >= 11 else 8
    return profile


# Bumped whenever an exported graph changes, so cached engines / ONNX files
# built from an older graph are not silently reused.
EXPORT_ARCH_TAG = "u3"


_EXPORT_CLASSES = {}


def _loadExport(interpolateMethod, ensemble=False):
    if interpolateMethod == "rife_elexor-tensorrt" and ensemble:
        from src.rifearches.IFNet_elexor_tensorrt import IFNet

        return IFNet, True
    for suffix in ("-directml", "-openvino"):
        if interpolateMethod.endswith(suffix):
            base = interpolateMethod[: -len(suffix)]
            if base in _DML_LEGACY:
                from src.rifearches import Rife_directml

                return getattr(Rife_directml, _DML_LEGACY[base]), base != "rife4.6"
    for suffix, wired in _EXPORT_WIRED.items():
        if interpolateMethod.endswith(suffix):
            base = interpolateMethod[: -len(suffix)]
            if base in wired:
                from src.rifearches import rife_fast

                fastCls = getattr(rife_fast, _EXPORT_FAST[base])
                key = (_EXPORT_FAST[base], suffix)
                cls = _EXPORT_CLASSES.get(key)
                if cls is None:
                    cls = rife_fast.exportArch(
                        fastCls,
                        name=f"Export{_EXPORT_FAST[base]}",
                        **_tensorrtProfile(),
                    )
                    cls.archTag = EXPORT_ARCH_TAG
                    _EXPORT_CLASSES[key] = cls
                return cls, fastCls.cfg["hasHead"]
    # Without this a new CLI choice that nobody added an arm for surfaces as
    # an obscure failure deep in handleModel instead of naming the method.
    raise ValueError(
        f"No v3 RIFE architecture is wired up for interpolate "
        f"method '{interpolateMethod}'."
    )


def importRifeArch(
    interpolateMethod, version, half=True, staticStep=False, ensemble=False
):
    match version:
        case "v1":
            return _loadV1(interpolateMethod, half, staticStep, ensemble)

        case "v3":
            return _loadExport(interpolateMethod, ensemble)
