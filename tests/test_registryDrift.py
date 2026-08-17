"""Drift guards between the hand-maintained registries.

Adding a model takes TWO edits (CLAUDE.md): the weight mapping in
``registry.modelsMap()`` AND the flag choices in the CLI parser.
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
from src.model.registry import modelsList, modelsMap


@pytest.fixture(scope="module")
def methodChoices():
    return capabilityMethods(_buildParser("."))


# --------------------------------------------------------------------------- #
# encode_method choices <-> matchEncoder arms
# --------------------------------------------------------------------------- #


def testEveryEncodeChoiceHasEncoderFlags(methodChoices):
    # *_nelux methods never reach matchEncoder: createWriteBuffer routes them to
    # NeluxWriteBuffer, which maps them to native encoder settings, and any that
    # still land on the FFmpeg writer are rewritten to their non-nelux twin by
    # WriteBuffer.__init__ before matchEncoder is called. Both twins are in this
    # list, so the carve-out costs no coverage.
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
# These would crash any code path that feeds them to modelsMap directly. Frozen
# so the set can only shrink; add a modelsMap arm instead of extending this.
KNOWN_MISSING_MODELSMAP_ARM = {
    "smosr-openvino",
    "video_small_v2",
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
    "stabilize": {
        "classic",  # CPU ORB/LK feature tracking, no weights
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
    "smooth_dedup": {
        # same comparators as dedup, driven off --smooth_dedup_method
        "ssim",
        "mse",
        "ssim-cuda",
        "mse-cuda",
        "vmaf",
        "vmaf-cuda",
    },
    "interpolate": {
        # Bare aliases: absent from modelsList, but modelsMap resolves them to
        # 4.22 (see testEveryInterpolateChoiceResolves, which pins that).
        "rife-ncnn",
        "rife-tensorrt",
        "rife4.15-tensorrt",  # modelsMap arm exists; absent from modelsList
        "rife4.15-directml",  # rides the rife4.15 ONNX via suffix replace
        "rife4.15-openvino",  # rides the rife4.15 ONNX via suffix replace
        "distildrba-tensorrt",  # weights ship via the base distildrba entry
        "distildrba-lite-tensorrt",  # weights ship via the base distildrba-lite entry
    },
    "scenechange": {
        # streaming scene-cut detectors; the cheap tier is algorithmic (no
        # weights). maxxvit-* ride the registered maxxvit ONNX entries.
        "ssim",
        "mse",
        "ssim-cuda",
        "mse-cuda",
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
        "Register the model in registry.modelsList()/modelsMap() or, if it "
        "genuinely needs no weights, add it to KNOWN_UNREGISTERED_METHODS with a reason."
    )
    assert stale == {}, (
        f"KNOWN_UNREGISTERED_METHODS entries no longer in the CLI choices: {stale}. "
        "Remove them from the exception table."
    )


# --------------------------------------------------------------------------- #
# --interpolate_method choices <-> what the backend actually resolves
# --------------------------------------------------------------------------- #

# The guards above compare choices against modelsList(), which the runtime never
# calls. The backends call modelsMap() and importRifeArch() with their own
# per-backend name mangling, and those two can disagree with each other. Three
# shipped choices were broken exactly there and none of the tests above saw it:
#
#   * --interpolate_method rife-ncnn      -> ValueError: Model rife-ncnn not found
#     (RifeNCNN passes the full method to modelsMap; only "rife" had an arm)
#   * --interpolate_method rife-tensorrt  -> UnboundLocalError: IFNet
#     (importRifeArch's v3 match had no arm and no default)
#   * --interpolate_method rife-mps       -> load_state_dict blows up
#     (arch table said 4.25, modelsMap said rife422.pth)
#
# So probe the real calls, and pin that methods sharing a weight file share an
# architecture — that pairing is what the third bug violated.


def _interpolateResolution(method):
    """(family, weightFile, archClass) exactly as the matching backend resolves.

    ``family`` separates the three architecture sets — the fp16 ``rife_fast``
    classes (CUDA/MPS), the TensorRT v3 modules and the DirectML/ORT v3 modules
    — because one ``.pth`` is legitimately loaded by one class from each.
    """
    from src.interpolate._shared import importRifeArch

    if method.endswith("-ncnn"):
        return "ncnn", modelsMap(method, modelType="ncnn"), None
    if method.endswith("-tensorrt"):
        base = method.replace("-tensorrt", "")
        return (
            "v3-trt",
            modelsMap(base, modelType="pth"),
            importRifeArch(method, "v3")[0],
        )
    if method.endswith(("-directml", "-openvino")):
        # RifeDirectML strips only "-directml"; OpenVINO rides the same arm.
        base = method.replace("-directml", "")
        return (
            "v3-ort",
            modelsMap(base, modelType="pth"),
            importRifeArch(method, "v3")[0],
        )
    base = method.replace("-mps", "")  # RifeMPS.baseMethod; CUDA uses it as-is
    return "v1", modelsMap(base, modelType="pth"), importRifeArch(base, "v1", half=True)


def testEveryInterpolateChoiceResolves(methodChoices):
    pytest.importorskip("torch")

    failures = {}
    resolved = {}
    for method in methodChoices["interpolate"]:
        if "drba" in method or method == "gmfss":
            continue  # own resolution path, not the rife arch table
        try:
            resolved[method] = _interpolateResolution(method)
        except Exception as e:  # noqa: BLE001 - any failure here is the bug
            failures[method] = f"{type(e).__name__}: {e}"

    assert failures == {}, (
        f"--interpolate_method choices that crash before inference: {failures}. "
        "Add the modelsMap arm and/or the importRifeArch case for the method."
    )

    # Same weights => same architecture. A bare alias pointing at one version's
    # arch while modelsMap hands it another version's .pth loads nothing.
    # Report the methods, not the classes: rife_fast builds its fp16 archs in a
    # factory, so they all share the name "_Concrete" and naming them says
    # nothing about which flag to fix.
    byWeight = {}
    for method, (family, weightFile, arch) in resolved.items():
        if arch is None:
            continue
        byWeight.setdefault((family, weightFile), {}).setdefault(arch, []).append(
            method
        )
    mismatched = {
        f"{family}/{weightFile}": sorted(sorted(m) for m in archs.values())
        for (family, weightFile), archs in byWeight.items()
        if len(archs) > 1
    }
    assert mismatched == {}, (
        f"Methods sharing a weight file but not an architecture (one group per "
        f"architecture): {mismatched}. load_state_dict would reject the "
        f"mismatched pair."
    )


def testUnwiredV3MethodNamesItself():
    # The v3 match must raise a named error, not UnboundLocalError.
    pytest.importorskip("torch")
    from src.interpolate._shared import importRifeArch

    with pytest.raises(ValueError, match="rife9.9-tensorrt"):
        importRifeArch("rife9.9-tensorrt", "v3")


# --------------------------------------------------------------------------- #
# --ensemble is a second axis through modelsMap, and nothing exercised it: the
# resolution helpers above always take the ensemble=False default. The
# rife4.22-lite ncnn arm returned `rife-v4.22-lite-ensemble-ncnn.zip`, a name
# that was never published, so `--interpolate_method rife4.22-lite-ncnn
# --ensemble` 404'd three times and killed the run -- while the sibling 4.22 and
# 4.21 arms already fell back to their non-ensemble zip.
# --------------------------------------------------------------------------- #


def _ncnnChoices(methodChoices):
    return [m for m in methodChoices["interpolate"] if m.endswith("-ncnn")]


def testEveryNcnnEnsembleChoiceResolvesToAName(methodChoices):
    # The blunt guard: an arm that falls off the end of its `case` returns None,
    # and a None filename reaches the URL join as the string "None".
    unresolved = {
        method: modelsMap(method, modelType="ncnn", ensemble=True)
        for method in _ncnnChoices(methodChoices)
    }

    assert {m: v for m, v in unresolved.items() if not v} == {}


def testNcnnEnsembleFallsBackToAPublishedModel(methodChoices):
    # Ensemble is unsupported past rife 4.21 -- the warning says so -- so those
    # arms must degrade to the non-ensemble zip rather than name one that does
    # not exist. Pinned against the non-ensemble name so the two cannot diverge.
    for method in _ncnnChoices(methodChoices):
        plain = modelsMap(method, modelType="ncnn", ensemble=False)
        ensemble = modelsMap(method, modelType="ncnn", ensemble=True)

        # "ensenmble" is not a typo here: the 4.15-lite asset is misspelled on
        # the model host itself (rife-v4.15-lite-ensenmble-ncnn.zip), so the
        # registry has to ask for that exact name. Correcting the spelling in
        # the code would 404.
        if "ensemble" in str(ensemble) or "ensenmble" in str(ensemble):
            # Only the arms that genuinely ship an ensemble zip may name one.
            assert ensemble != plain, method
        else:
            assert ensemble == plain, (
                f"{method} with --ensemble resolves to {ensemble!r}, which is "
                f"neither an ensemble build nor the non-ensemble {plain!r}."
            )


def testRife422LiteNcnnEnsembleUsesTheNonEnsembleZip():
    # The specific regression, named so its fix cannot be quietly reverted.
    assert (
        modelsMap("rife4.22-lite-ncnn", modelType="ncnn", ensemble=True)
        == "rife-v4.22-lite-ncnn.zip"
    )
