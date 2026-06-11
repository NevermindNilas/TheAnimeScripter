"""Tests for src/utils/inputOutputHandler.py — output naming and path resolution.

Covers the I/O overhaul: collision-safe naming, Windows sequence de-duplication,
deterministic batch order, extension/feature resolution for URLs and sequences,
and unsafe-character scrubbing.
"""

import os
import types

import pytest

from src.io import inputOutputHandler as io


def make_args(**overrides):
    """Minimal args namespace with every attribute generateOutputName reads."""
    base = dict(
        resize=0, resize_factor=2,
        dedup=0, dedup_sens=50,
        interpolate=0, interpolate_factor=2,
        upscale=0, upscale_factor=2,
        restore=0, restore_method="scunet",
        segment=0, depth=0, ytdlp=0,
        single_image_input=0,
        encode_method="", png_passthrough=0,
        custom_encoder=None,
        output=None, input=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# generateOutputName: base name, extension, feature suffixes, sanitisation
# --------------------------------------------------------------------------- #

def test_extension_copied_from_input_file():
    assert io.generateOutputName(make_args(upscale=1), "C:/v/clip.mkv") == "clip-Up2.mkv"


def test_prores_forces_mov():
    assert io.generateOutputName(make_args(encode_method="prores"), "clip.mp4") == "clip.mov"


def test_segment_forces_mov():
    assert io.generateOutputName(make_args(segment=1), "clip.mp4") == "clip-Segment.mov"


def test_single_image_forces_png():
    assert io.generateOutputName(make_args(single_image_input=1), "clip.mp4") == "clip.png"


def test_no_input_falls_back_to_tas():
    assert io.generateOutputName(make_args(), None) == "TAS.mp4"


def test_resize_suffix_carries_factor():
    assert io.generateOutputName(make_args(resize=1), "clip.mkv") == "clip-Resize2.mkv"


def test_dedup_suffix_carries_sensitivity():
    assert io.generateOutputName(make_args(dedup=1), "clip.mkv") == "clip-Dedup50.mkv"


def test_feature_suffix_order_is_stable():
    # Suffix order is fixed by the features table, not by flag order:
    # Resize, Dedup, Int, Up, Restore, ...
    name = io.generateOutputName(
        make_args(resize=1, dedup=1, interpolate=1, upscale=1), "clip.mkv"
    )
    assert name == "clip-Resize2-Dedup50-Int2-Up2.mkv"


# ---- #5 URL respects encode_method + feature suffixes ---------------------- #

@pytest.mark.parametrize("url", ["https://youtu.be/x", "http://y/z"])
def test_url_default_extension(url):
    assert io.generateOutputName(make_args(), url) == "TAS-YTDLP.mp4"


def test_url_keeps_feature_suffix():
    assert io.generateOutputName(make_args(interpolate=1), "https://y/z") == "TAS-YTDLP-Int2.mp4"


def test_url_respects_prores():
    assert io.generateOutputName(make_args(encode_method="prores"), "https://y/z") == "TAS-YTDLP.mov"


def test_url_no_duplicate_ytdlp_tag():
    # ytdlp flag set on a URL must not produce TAS-YTDLP-YTDLP
    assert io.generateOutputName(make_args(ytdlp=1), "https://y/z") == "TAS-YTDLP.mp4"


def test_local_file_still_gets_ytdlp_tag():
    assert io.generateOutputName(make_args(ytdlp=1), "C:/v/clip.mp4") == "clip-YTDLP.mp4"


# ---- #6 image-sequence input: strip %05d, no garbage extension ------------- #

def test_sequence_input_strips_pattern_and_picks_container():
    assert io.generateOutputName(make_args(encode_method="prores"), "C:/v/frames_%05d.png") == "frames.mov"


def test_sequence_input_default_extension():
    assert io.generateOutputName(make_args(), "C:/v/frames_%05d.png") == "frames.mp4"


# ---- #9 unsafe characters scrubbed ---------------------------------------- #

def test_unsafe_chars_sanitised():
    name = io.generateOutputName(make_args(restore=1, restore_method="a/b:c*?"), "clip.mp4")
    assert name == "clip-Restorea_b_c__.mp4"


# --------------------------------------------------------------------------- #
# #1 Windows duplicate-glob: image sequence detection
# --------------------------------------------------------------------------- #

def test_sequence_detection_counts_each_frame_once(tmp_path):
    for i in range(1, 6):
        (tmp_path / f"frame_{i:03d}.png").touch()
    seq = io.detectImageSequence(str(tmp_path))
    assert seq is not None
    pattern, first, last, count = seq
    assert pattern.endswith("frame_%03d.png")
    assert (first, last, count) == (1, 5, 5)  # 5, not 10 (case-insensitive glob)


def test_sequence_detection_none_for_single_image(tmp_path):
    (tmp_path / "frame_001.png").touch()
    assert io.detectImageSequence(str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# generateOutputPath: collision-safe naming (#2 #3 #4 + explicit output)
# --------------------------------------------------------------------------- #

def test_batch_same_basename_no_clobber(tmp_path):
    out = str(tmp_path)
    used = set()
    p1 = io.generateOutputPath("A/clip.mp4", None, out, make_args(), used)
    p2 = io.generateOutputPath("B/clip.mp4", None, out, make_args(), used)
    assert p1 != p2
    assert os.path.basename(p1) == "clip.mp4"
    assert os.path.basename(p2) == "clip-1.mp4"


def test_existing_file_on_disk_is_bumped(tmp_path):
    out = str(tmp_path)
    (tmp_path / "clip.mp4").touch()
    used = set()
    p = io.generateOutputPath("X/clip.mp4", None, out, make_args(), used)
    assert os.path.basename(p) == "clip-1.mp4"


def test_explicit_output_file_overwritable_for_single(tmp_path):
    out = str(tmp_path)
    explicit = str(tmp_path / "final.mp4")
    (tmp_path / "final.mp4").touch()  # already exists -> still reused
    used = set()
    p = io.generateOutputPath("X/a.mp4", explicit, out, make_args(output=explicit), used)
    assert p == explicit


def test_explicit_output_file_disambiguated_in_batch(tmp_path):
    out = str(tmp_path)
    explicit = str(tmp_path / "final.mp4")
    used = set()
    p1 = io.generateOutputPath("A/a.mp4", explicit, out, make_args(output=explicit), used)
    p2 = io.generateOutputPath("B/b.mp4", explicit, out, make_args(output=explicit), used)
    assert os.path.basename(p1) == "final.mp4"
    assert os.path.basename(p2) == "final-1.mp4"


# ---- #11 png sequence output -> unique folder ------------------------------ #

def test_png_sequence_creates_folder(tmp_path):
    out = str(tmp_path)
    used = set()
    p = io.generateOutputPath("X/clip.mp4", None, out, make_args(encode_method="png"), used)
    assert p.endswith(os.path.join("clip", "frames_%05d.png"))
    assert os.path.isdir(os.path.dirname(p))


def test_png_sequence_folder_collision_bumped(tmp_path):
    out = str(tmp_path)
    used = set()
    io.generateOutputPath("X/clip.mp4", None, out, make_args(encode_method="png"), used)
    p2 = io.generateOutputPath("Y/clip.mp4", None, out, make_args(encode_method="png"), used)
    assert "clip-1" in p2


def test_png_passthrough_stays_a_file_not_folder(tmp_path):
    out = str(tmp_path)
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4", None, out,
        make_args(encode_method="png", png_passthrough=1, single_image_input=1),
        used,
    )
    assert p.endswith(".png")
    assert "frames_%05d" not in p


# --------------------------------------------------------------------------- #
# getVideoFiles + processInputOutputPaths (#7 sorted, #8 list)
# --------------------------------------------------------------------------- #

def test_directory_listing_is_sorted(tmp_path):
    for nm in ["c.mp4", "a.mp4", "b.mp4"]:
        (tmp_path / nm).touch()
    files = io.getVideoFiles(str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["a.mp4", "b.mp4", "c.mp4"]


def test_semicolon_separated_inputs(tmp_path):
    a = tmp_path / "a.mp4"; a.touch()
    b = tmp_path / "b.mp4"; b.touch()
    files = io.getVideoFiles(f"{a};{b}")
    assert sorted(os.path.basename(f) for f in files) == ["a.mp4", "b.mp4"]


def test_url_passed_through():
    assert io.getVideoFiles("https://youtu.be/x") == ["https://youtu.be/x"]


def test_sequence_pattern_passed_through():
    assert io.getVideoFiles("C:/v/frames_%05d.png") == ["C:/v/frames_%05d.png"]


def test_process_returns_sorted_unique_list(tmp_path):
    vdir = tmp_path / "vids"; vdir.mkdir()
    out = tmp_path / "out"
    for nm in ["c.mp4", "a.mp4", "b.mp4"]:
        (vdir / nm).touch()
    res = io.processInputOutputPaths(make_args(input=str(vdir)), str(out))
    assert isinstance(res, list) and len(res) == 3
    assert [os.path.basename(r["videoPath"]) for r in res] == ["a.mp4", "b.mp4", "c.mp4"]
    assert len({r["outputPath"] for r in res}) == 3  # all unique


def test_process_txt_list_input(tmp_path):
    vdir = tmp_path / "vids"; vdir.mkdir()
    a = vdir / "a.mp4"; a.touch()
    b = vdir / "b.mp4"; b.touch()
    lst = tmp_path / "list.txt"
    lst.write_text(f'"{a}"\n{b}\n')
    res = io.processInputOutputPaths(make_args(input=str(lst)), str(tmp_path / "out"))
    assert len(res) == 2


def test_process_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.processInputOutputPaths(make_args(input=str(tmp_path / "nope.mp4")), str(tmp_path / "out"))


# --------------------------------------------------------------------------- #
# #13 validateEncoder webm handling
# --------------------------------------------------------------------------- #

def test_webm_without_compatible_encoder_falls_back_to_vp9():
    assert io.validateEncoder("x.webm", "h264", None) == "vp9"


def test_webm_with_custom_encoder_kept():
    assert io.validateEncoder("x.webm", "h264", "-c:v libx264") == "h264"


def test_webm_with_compatible_encoder_kept():
    assert io.validateEncoder("x.webm", "nvenc_av1", None) == "nvenc_av1"


def test_non_webm_untouched():
    assert io.validateEncoder("x.mp4", "h264", None) == "h264"
