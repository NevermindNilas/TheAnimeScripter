"""Drift guards between the hand-maintained registries.

Adding a model takes TWO edits (CLAUDE.md): the weight mapping in
``downloadModels.modelsMap()`` AND the flag choices in ``argumentsChecker``.
Same story for encoders: the argparse ``encode_method`` choices and the
``match`` arms in ``encodingSettings.matchEncoder()``. Nothing enforced these
until now, and both had already drifted when these tests were written:

  * ``--encode_method lossless_nvenc`` was an advertised choice but
    ``matchEncoder`` only had a ``lossless_nvenc_h264`` arm (unreachable from
    the CLI), so the choice silently produced an FFmpeg command with no -c:v.
  * ``--interpolate_method rife4.16-lite`` was wired through the CLI,
    initializeModels and the arch table, but had no ``modelsMap`` arm, so it
    crashed with ``ValueError: Model rife4.16-lite not found.`` at download.

The choice<->registry comparison normalises the backend aliasing the runtime
performs (OpenVINO rides the DirectML/ORT path, rife DirectML/OpenVINO reuse
the TensorRT ONNX, depth strips the ``og_`` prefix, ``-mps`` is stripped
before modelsMap). What can't be normalised is frozen in per-capability
exception sets: methods that need no downloadable weights (algorithmic, SDKs)
or that alias another registry name. A NEW method that isn't registered and
isn't excepted fails the test — which is exactly the "forgot the second edit"
mistake these guards exist to catch.
"""

import pytest

from src.cli.parser import _buildParser, capabilityMethods
from src.io.encodingSettings import matchEncoder
from src.model.downloadModels import modelsList, modelsMap


@pytest.fixture(scope="module")
def methodChoices():
    return capabilityMethods(_buildParser("."))


# --------------------------------------------------------------------------- #
# encode_method choices <-> matchEncoder arms
# --------------------------------------------------------------------------- #


def testEveryEncodeChoiceHasEncoderFlags(methodChoices):
    # *_nelux methods never reach matchEncoder; NeluxWriteBuffer maps them to
    # native encoder settings in ffmpegSettings instead.
    missing = [
        m
        for m in methodChoices["encode"]
        if not m.endswith("_nelux") and matchEncoder(m) == []
    ]
    assert missing == [], (
        f"encode_method choices without a matchEncoder arm: {missing}. "
        "FFmpeg would be invoked without -c:v and silently pick a default codec."
    )


def testUnknownEncoderStillReturnsEmpty():
    # The guard above relies on [] meaning "no arm"; pin that contract.
    assert matchEncoder("definitely_not_an_encoder") == []


# --------------------------------------------------------------------------- #
# modelsList() internal consistency
# --------------------------------------------------------------------------- #


def testModelsListHasNoDuplicates():
    names = modelsList()
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert dupes == []


# Registry entries with no modelsMap arm even after the runtime's -mps strip.
# These are reachable from --offline all (failure is caught + logged there) but
# would crash any code path that feeds them to modelsMap directly. Frozen so
# the set can only shrink; add a modelsMap arm instead of extending this.
KNOWN_MISSING_MODELSMAP_ARM = {
    "smosr-openvino",
    "video_small_v2",
    "video_large_v2",
    "yolov9_small_mit",
    "yolov9_medium_mit",
    "yolov9_large_mit",
}


def _modelTypeFor(name):
    if name.endswith("-ncnn"):
        return "ncnn"
    if name.endswith(("-tensorrt", "-directml", "-openvino")):
        return "onnx"
    return "pth"


def testEveryModelsListEntryHasModelsMapArm():
    # half=False because some arms (scunet onnx) deliberately reject fp16.
    # The runtime strips "-mps" before calling modelsMap (adjustMethod /
    # RifeMPS.baseMethod), so probe the stripped name.
    missing = set()
    for name in modelsList():
        probe = name[: -len("-mps")] if name.endswith("-mps") else name
        try:
            modelsMap(probe, modelType=_modelTypeFor(probe), half=False)
        except ValueError as e:
            if "not found" in str(e).lower():
                missing.add(name)

    new = missing - KNOWN_MISSING_MODELSMAP_ARM
    fixed = KNOWN_MISSING_MODELSMAP_ARM - missing
    assert new == set(), (
        f"modelsList entries without a modelsMap arm: {sorted(new)}. "
        "Add the weight mapping (CLAUDE.md: adding a model = TWO edits)."
    )
    assert fixed == set(), (
        f"These now resolve fine — remove them from KNOWN_MISSING_MODELSMAP_ARM: "
        f"{sorted(fixed)}"
    )


# --------------------------------------------------------------------------- #
# capability method choices <-> modelsList() registry
# --------------------------------------------------------------------------- #

# Methods that legitimately have no modelsList entry. Grouped by capability so
# a failure points at the right choices list. Keep the reason next to the name.
KNOWN_UNREGISTERED_METHODS = {
    "autoclip": {
        "pyscenedetect",  # CPU PySceneDetect, no weights
    },
    "dedup": {
        # algorithmic comparators, no weights (only flownets downloads)
        "ssim",
        "mse",
        "ssim-cuda",
        "mse-cuda",
        "vmaf",
        "vmaf-cuda",
    },
    "interpolate": {
        "rife-ncnn",  # generic alias resolved to a default version
        "rife-tensorrt",  # generic alias resolved to a default version
        "rife4.15-tensorrt",  # modelsMap arm exists; absent from modelsList (--offline gap)
        "rife4.15-directml",  # rides the rife4.15 ONNX via suffix replace
        "rife4.15-openvino",  # rides the rife4.15 ONNX via suffix replace
        "distildrba-tensorrt",  # weights ship via the base distildrba entry
        "distildrba-lite-tensorrt",  # weights ship via the base distildrba-lite entry
    },
    "restore": {
        # registered under their -mps/-tensorrt siblings or external SDKs
        "fastlinedarken",
        "fastlinedarken-tensorrt",
        "autocas",  # sharpening kernel, no weights
        "deh264_real",
        "deh264_real-tensorrt",
        "deh264_real-directml",
        "deh264_real-openvino",
        "deh264_span",
        "deh264_span-tensorrt",
        "deh264_span-directml",
        "deh264_span-openvino",
        "linethinner-lite",
        "linethinner-medium",
        "linethinner-heavy",
        "linethinner-lite-cuda",
        "linethinner-medium-cuda",
        "linethinner-heavy-cuda",
        "maxine-denoise_low",
        "maxine-denoise_medium",
        "maxine-denoise_high",
        "maxine-denoise_ultra",
        "maxine-deblur_low",
        "maxine-deblur_medium",
        "maxine-deblur_high",
        "maxine-deblur_ultra",
    },
    "segment": {
        # choices use the user-facing names; weights live under "segment*"
        "anime",
        "anime-tensorrt",
        "anime-directml",
        "cartoon",
    },
    "upscale": {
        # NVIDIA Maxine SDK effects, no downloadable weights
        "maxine-bicubic",
        "maxine-low",
        "maxine-medium",
        "maxine-high",
        "maxine-ultra",
        "maxine-highbitrate_low",
        "maxine-highbitrate_medium",
        "maxine-highbitrate_high",
        "maxine-highbitrate_ultra",
    },
}


def _registeredVariants(method):
    """Names the runtime may resolve this method to before hitting the registry.

    Mirrors the backend aliasing conventions: OpenVINO is a branch inside the
    DirectML/ORT classes, rife/distildrba DirectML+OpenVINO reuse the TensorRT
    ONNX, and the depth backends strip the ``og_`` prefix.
    """
    variants = {method}
    variants.add(method.replace("-openvino", "-directml"))
    variants.add(method.replace("-openvino", "-tensorrt"))
    variants.add(method.replace("-directml", "-tensorrt"))
    variants |= {v.removeprefix("og_") for v in set(variants)}
    return variants


def testEveryMethodChoiceIsRegisteredOrKnownException(methodChoices):
    registry = set(modelsList())
    problems = {}
    stale = {}

    for capability, methods in methodChoices.items():
        if capability == "encode":  # guarded against matchEncoder above
            continue
        allowed = KNOWN_UNREGISTERED_METHODS.get(capability, set())
        unregistered = {m for m in methods if not (_registeredVariants(m) & registry)}
        new = sorted(unregistered - allowed)
        if new:
            problems[capability] = new
        gone = sorted(allowed - set(methods))
        if gone:
            stale[capability] = gone

    assert problems == {}, (
        f"Method choices with no modelsList entry (any backend variant): {problems}. "
        "Register the model in downloadModels.modelsList()/modelsMap() or, if it "
        "genuinely needs no weights, add it to KNOWN_UNREGISTERED_METHODS with a reason."
    )
    assert stale == {}, (
        f"KNOWN_UNREGISTERED_METHODS entries no longer in the CLI choices: {stale}. "
        "Remove them from the exception table."
    )
