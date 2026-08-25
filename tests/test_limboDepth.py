"""Guards for the Limbo depth method.

Limbo is the only depth method whose input resolution is fixed by the model
rather than by ``--depth_quality``: it ships one ONNX per baked size and the
backends pick between them from the source aspect ratio. Two things can drift
silently there -- the picker landing a 16:9 source on the 4:3 export (the model
still returns a plausible depth map, just a worse one), and a backend forgetting
that Limbo's export, unlike depth_anything_v2's, has no ImageNet normalization
baked into the graph. The second one shipped once during development: it cost
~19 dB against the ONNX reference and looked fine on screen.
"""

import ast
from pathlib import Path

import pytest

from src.model.registry import modelsMap

SRC = Path(__file__).resolve().parent.parent / "src"

WIDE = (280, 504)
FOUR_THREE = (378, 504)


# --------------------------------------------------------------------------- #
# resolution picker
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1920, 1080, WIDE),
        (1280, 720, WIDE),
        (3840, 2160, WIDE),
        (1920, 804, WIDE),  # 2.39:1 scope
        (1440, 1080, FOUR_THREE),
        (640, 480, FOUR_THREE),
        (720, 576, FOUR_THREE),  # 5:4-ish PAL
        (1024, 1024, FOUR_THREE),  # square
        (1080, 1920, FOUR_THREE),  # portrait: neither export is tall
    ],
)
def testLimboResolutionPicksTheNearerExport(width, height, expected):
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.depth.backends._shared import limboResolution

    assert limboResolution(width, height) == expected


def testLimboResolutionOnlyEverReturnsAnExportedShape():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.depth.backends._shared import LIMBO_SHAPES, limboResolution

    # Anything the picker returns has to be a shape a weight file actually
    # exists for, or the backends resolve a filename that 404s.
    for width in range(64, 4097, 137):
        for height in range(64, 4097, 149):
            assert limboResolution(width, height) in LIMBO_SHAPES


# --------------------------------------------------------------------------- #
# disparity post-processing
# --------------------------------------------------------------------------- #


def testLimboDisparityInvertsAndStretches():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.depth.backends._shared import limboDisparity

    # Limbo emits positive depth (further = larger), so the far pixel must come
    # out dark and the near one bright.
    depth = torch.linspace(1.0, 10.0, 64).reshape(1, 1, 8, 8)
    gray = limboDisparity(depth)

    assert gray.shape == depth.shape
    assert gray.min() >= 0.0 and gray.max() <= 1.0
    assert gray.flatten()[0] > gray.flatten()[-1]


def testLimboDisparityHandlesADegenerateFrame():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from src.depth.backends._shared import limboDisparity

    # All-invalid (and all-flat) frames must write black, not NaN: a NaN reaching
    # the writer's quantization poisons the encoded frame.
    for depth in (
        torch.zeros(1, 1, 8, 8),
        torch.full((1, 1, 8, 8), float("nan")),
        torch.full((1, 1, 8, 8), 3.0),
    ):
        gray = limboDisparity(depth)
        assert torch.isfinite(gray).all()
        assert gray.min() >= 0.0 and gray.max() <= 1.0


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def testBothLimboExportsAreRegistered():
    # The 4:3 arm has no CLI choice of its own -- only limboResolution reaches
    # it -- so nothing else in the drift suite would notice it disappearing.
    assert modelsMap("limbo", modelType="pth") == "Limbo.safetensors"
    assert modelsMap("limbo-tensorrt", modelType="onnx", half=True) == (
        "Limbo_504x280_fp16.onnx"
    )
    assert modelsMap("limbo-tensorrt", modelType="onnx", half=False) == (
        "Limbo_504x280_fp32.onnx"
    )
    assert modelsMap("limbo_43-tensorrt", modelType="onnx", half=True) == (
        "Limbo_504x378_fp16.onnx"
    )
    assert modelsMap("limbo_43-directml", modelType="onnx", half=False) == (
        "Limbo_504x378_fp32.onnx"
    )


def testLimboWeightNamesCarryTheResolutionTheyAreBuiltFor():
    # The filename is the only place the baked shape is written down, and the
    # backends trust it to match what limboResolution picked.
    for model, shape in (
        ("limbo-tensorrt", "504x280"),
        ("limbo_43-tensorrt", "504x378"),
    ):
        for half in (True, False):
            assert shape in modelsMap(model, modelType="onnx", half=half)


# --------------------------------------------------------------------------- #
# every Limbo backend normalizes its own input
# --------------------------------------------------------------------------- #

LIMBO_CLASSES = {
    "cuda.py": "LimboCuda",
    "mps.py": "LimboMPS",
    "tensorrt.py": "LimboTensorRT",
    "directml.py": "LimboOpenVino",
}


def _classNode(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return tree, node
    raise AssertionError(f"{name} is gone from {path.name}")


def _classBody(path, name):
    return ast.dump(_classNode(path, name)[1])


def _normalizes(path, name):
    """True if this class, or the base it inherits its frame prep from,
    references an ImageNet mean."""
    tree, node = _classNode(path, name)
    if "meanTensor" in ast.dump(node) or "MEANTENSOR" in ast.dump(node):
        return True
    for base in node.bases:
        if not isinstance(base, ast.Name):
            continue
        for other in ast.walk(tree):
            if isinstance(other, ast.ClassDef) and other.name == base.id:
                dumped = ast.dump(other)
                if "meanTensor" in dumped or "MEANTENSOR" in dumped:
                    return True
    return False


@pytest.mark.parametrize("filename,className", sorted(LIMBO_CLASSES.items()))
def testEveryLimboBackendNormalizesItsInput(filename, className):
    # AST-level because constructing these needs weights, a GPU, and for MPS
    # Apple hardware. Each backend must reach a mean/std tensor, in its own body
    # or in the base whose normFrame it reuses: the ONNX has no normalization
    # node, so an unnormalized frame is silently wrong rather than an error.
    # DepthDirectMLV2 is the one base that does NOT normalize (the v2 ONNX bakes
    # it in), which is exactly the inheritance that produced the bug.
    assert _normalizes(SRC / "depth" / "backends" / filename, className), (
        f"{className} never touches the ImageNet mean, and neither does the "
        f"class it inherits from. Limbo's export does not normalize internally "
        f"-- depth_anything_v2's does, which is why some of these bases get "
        f"away without it."
    )


@pytest.mark.parametrize("filename,className", sorted(LIMBO_CLASSES.items()))
def testEveryLimboBackendUsesTheSharedResolutionPicker(filename, className):
    body = _classBody(SRC / "depth" / "backends" / filename, className)
    assert "limboResolution" in body, (
        f"{className} does not call limboResolution, so it can size itself for "
        f"one export and download the other."
    )
    assert "calculateAspectRatio" not in body, (
        f"{className} calls calculateAspectRatio, which derives the resolution "
        f"from --depth_quality. Limbo's is baked into the weight file."
    )


def testEveryLimboCliChoiceHasAFactoryArm():
    from src.cli.parser import _buildParser, capabilityMethods

    methods = [m for m in capabilityMethods(_buildParser("."))["depth"] if "limbo" in m]
    assert sorted(methods) == [
        "limbo",
        "limbo-mps",
        "limbo-openvino",
        "limbo-tensorrt",
    ]

    source = (SRC / "factories" / "standalone.py").read_text(encoding="utf-8")
    for method in methods:
        assert f'case "{method}"' in source, (
            f"--depth_method {method} is an accepted CLI choice with no arm in "
            f"standalone.depth(); it would raise at runtime."
        )
