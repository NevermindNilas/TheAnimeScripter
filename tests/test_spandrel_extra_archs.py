"""Round-trip tests for the in-repo spandrel arches added on top of the fork:
DIS, FIGSR, GateRV3.

The contract for "an arch is included in spandrel" is: given a state_dict produced
by the arch module, MAIN_REGISTRY must (1) detect it as that arch and (2) rebuild a
model whose `load_state_dict(..., strict=True)` succeeds — which only happens if
every hyperparameter was inferred exactly. ModelLoader.load_from_state_dict triggers
that strict load internally, so a successful load IS the proof of correct inference.
A forward pass on a tiny input is asserted on top for output-shape parity.

Configs are deliberately non-default so the inference logic is exercised rather than
just matching constructor defaults.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")  # FIGSR arch module imports einops
pytest.importorskip("numpy")

from src.spandrelCompat import ModelLoader  # noqa: E402


def _roundtrip(model, expected_id, expected_scale, in_ch, out_ch):
    model.eval()
    sd = model.state_dict()
    desc = ModelLoader().load_from_state_dict(sd)  # strict load_state_dict inside
    assert desc.architecture.id == expected_id
    assert desc.scale == expected_scale
    assert desc.input_channels == in_ch
    assert desc.output_channels == out_ch
    return desc


def _assert_forward(desc, in_ch, scale, size=16):
    x = torch.rand(1, in_ch, size, size)
    with torch.no_grad():
        y = desc.model(x)
    assert y.shape[0] == 1
    assert y.shape[2] == size * scale
    assert y.shape[3] == size * scale


# --------------------------------------------------------------------------- #
# DIS — Direct Image Supersampling (ported from chaiNNer-org mainstream)
# --------------------------------------------------------------------------- #


def testDISFastResBlockSR():
    from src.extraArches.DIS import DIS

    model = DIS(num_features=16, num_blocks=4, scale=2, use_depthwise=False)
    desc = _roundtrip(model, "DIS", 2, 3, 3)
    assert desc.purpose == "SR"
    _assert_forward(desc, 3, 2)


def testDISDepthwiseScale4():
    from src.extraArches.DIS import DIS

    model = DIS(num_features=24, num_blocks=8, scale=4, use_depthwise=True)
    desc = _roundtrip(model, "DIS", 4, 3, 3)
    _assert_forward(desc, 3, 4)


def testDISRestorationScale1():
    from src.extraArches.DIS import DIS

    model = DIS(num_features=16, num_blocks=3, scale=1, use_depthwise=False)
    desc = _roundtrip(model, "DIS", 1, 3, 3)
    assert desc.purpose == "Restoration"
    _assert_forward(desc, 3, 1)


# --------------------------------------------------------------------------- #
# FIGSR — Fourier Inception Gated SR
# --------------------------------------------------------------------------- #


def testFIGSRPixelShuffleDirect():
    from src.extraArches.figsr import FIGSR

    model = FIGSR(
        dim=24,
        n_blocks=6,
        scale=2,
        upsampler="pixelshuffledirect",
        mid_dim=16,
        gc=4,
        square_kernel_size=7,
        band_kernel_size=9,
    )
    desc = _roundtrip(model, "FIGSR", 2, 3, 3)
    _assert_forward(desc, 3, 2)


# --------------------------------------------------------------------------- #
# SMoSR — Spatial Modulation SR (Umzi). Reparameterizable: .eval() fuses the
# ConvNXC/DOConv2d branches, so the detector must round-trip both rep variants.
# --------------------------------------------------------------------------- #


def testSMoSRRepPixelShuffleDirect():
    from spandrel.architectures.SMoSR import SMoSR

    model = SMoSR(dim=24, n_mb=2, scale=2, rep=True)
    desc = _roundtrip(model, "SMoSR", 2, 3, 3)
    assert desc.purpose == "SR"
    assert "rep" in desc.tags
    _assert_forward(desc, 3, 2)


def testSMoSRNoRep():
    from spandrel.architectures.SMoSR import SMoSR

    model = SMoSR(dim=16, n_mb=3, scale=2, rep=False)
    desc = _roundtrip(model, "SMoSR", 2, 3, 3)
    assert "norep" in desc.tags
    _assert_forward(desc, 3, 2)


# --------------------------------------------------------------------------- #
# GateRV3
# --------------------------------------------------------------------------- #


def testGateRV3SR():
    from src.extraArches.gaterv3 import GateRV3

    model = GateRV3(
        dim=16,
        enc_blocks=(1, 1),
        dec_blocks=(1, 1),
        num_latent=1,
        scale=2,
        upsample="pixelshuffledirect",
        upsample_mid_dim=16,
    )
    desc = _roundtrip(model, "GateRV3", 2, 3, 3)
    _assert_forward(desc, 3, 2)


def testGateRV3RestorationScale1():
    from src.extraArches.gaterv3 import GateRV3

    model = GateRV3(
        dim=16,
        enc_blocks=(1, 1),
        dec_blocks=(1, 1),
        num_latent=1,
        scale=1,
    )
    desc = _roundtrip(model, "GateRV3", 1, 3, 3)
    assert desc.purpose == "Restoration"
    _assert_forward(desc, 3, 1)
