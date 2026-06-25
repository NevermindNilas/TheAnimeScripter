import os
import types

import pytest

import src.constants as cs
from src.io.inputNormalization import InputNormalizationError, normalizeInputArgs


def make_args(**overrides):
    base = dict(
        input="clip.mp4",
        output=None,
        encode_method="x264",
        custom_encoder=None,
        png_passthrough=False,
        single_image_input=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def testNormalFileInputBecomesAbsolute(tmp_path):
    video = tmp_path / "clip.mp4"
    args = make_args(input=str(video))

    assert normalizeInputArgs(args, str(tmp_path), processingEnabled=False) is True

    assert args.input == str(video.resolve())


def testMissingInputRaises():
    args = make_args(input=None)

    with pytest.raises(InputNormalizationError, match="No input specified"):
        normalizeInputArgs(args, ".", processingEnabled=False)


def testImageSequencePatternBecomesAbsoluteAndDisablesAudio(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "AUDIO", True, raising=False)
    pattern = tmp_path / "frames_%05d.png"
    args = make_args(input=str(pattern))

    assert normalizeInputArgs(args, str(tmp_path), processingEnabled=False) is True

    assert args.input == os.path.abspath(pattern)
    assert cs.AUDIO is False


def testSinglePngEnablesPassthroughAndDisablesAudio(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "AUDIO", True, raising=False)
    image = tmp_path / "frame.png"
    args = make_args(input=str(image))

    assert normalizeInputArgs(args, str(tmp_path), processingEnabled=False) is True

    assert args.input == str(image.resolve())
    assert args.single_image_input is True
    assert args.png_passthrough is True
    assert cs.AUDIO is False
    assert args.encode_method == "x264"


def testSinglePngWithProcessingForcesPngEncode(tmp_path):
    image = tmp_path / "frame.png"
    args = make_args(input=str(image), encode_method="x264")

    normalizeInputArgs(args, str(tmp_path), processingEnabled=True)

    assert args.encode_method == "png"


def testSingleJpegRaises(tmp_path):
    image = tmp_path / "frame.jpg"
    args = make_args(input=str(image))

    with pytest.raises(InputNormalizationError, match="Single image input"):
        normalizeInputArgs(args, str(tmp_path), processingEnabled=False)


def testGifInputForcesGifEncode(tmp_path):
    image = tmp_path / "clip.gif"
    args = make_args(input=str(image), encode_method="x264")

    normalizeInputArgs(args, str(tmp_path), processingEnabled=False)

    assert args.encode_method == "gif"
