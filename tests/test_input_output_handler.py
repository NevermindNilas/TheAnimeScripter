"""Tests for src.io.inputOutputHandler — output naming and path resolution.

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
        resize=0,
        resize_factor=2,
        dedup=0,
        dedup_sens=50,
        interpolate=0,
        interpolate_factor=2,
        upscale=0,
        upscale_factor=2,
        restore=0,
        restore_method="scunet",
        segment=0,
        depth=0,
        ytdlp=0,
        single_image_input=0,
        encode_method="",
        png_passthrough=0,
        custom_encoder=None,
        output=None,
        input=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# generateOutputName: base name, extension, feature suffixes, sanitisation
# --------------------------------------------------------------------------- #


def test_extension_copied_from_input_file():
    assert (
        io.generateOutputName(make_args(upscale=1), "C:/v/clip.mkv") == "clip-Up2.mkv"
    )


def test_prores_forces_mov():
    assert (
        io.generateOutputName(make_args(encode_method="prores"), "clip.mp4")
        == "clip.mov"
    )


def test_segment_forces_mov():
    assert io.generateOutputName(make_args(segment=1), "clip.mp4") == "clip-Segment.mov"


def test_prores_nelux_forces_mov_like_its_twin():
    assert (
        io.generateOutputName(make_args(encode_method="prores_nelux"), "clip.mp4")
        == "clip.mov"
    )


def test_gif_nelux_resolves_gif_extension_like_its_twin():
    # gif into the input's container fails at header write either way; the
    # nelux twin needs the same .gif routing as `-c:v gif`.
    assert (
        io.generateOutputName(make_args(encode_method="gif_nelux"), "clip.mp4")
        == "clip.gif"
    )


def test_single_image_forces_png():
    assert (
        io.generateOutputName(make_args(single_image_input=1), "clip.mp4") == "clip.png"
    )


def test_single_image_honours_jpeg():
    assert (
        io.generateOutputName(
            make_args(single_image_input=1, encode_method="jpeg"), "clip.mp4"
        )
        == "clip.jpg"
    )


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
    assert (
        io.generateOutputName(make_args(interpolate=1), "https://y/z")
        == "TAS-YTDLP-Int2.mp4"
    )


def test_url_respects_prores():
    assert (
        io.generateOutputName(make_args(encode_method="prores"), "https://y/z")
        == "TAS-YTDLP.mov"
    )


def test_url_no_duplicate_ytdlp_tag():
    # ytdlp flag set on a URL must not produce TAS-YTDLP-YTDLP
    assert io.generateOutputName(make_args(ytdlp=1), "https://y/z") == "TAS-YTDLP.mp4"


def test_local_file_still_gets_ytdlp_tag():
    assert (
        io.generateOutputName(make_args(ytdlp=1), "C:/v/clip.mp4") == "clip-YTDLP.mp4"
    )


# ---- #6 image-sequence input: strip %05d, no garbage extension ------------- #


def test_sequence_input_strips_pattern_and_picks_container():
    assert (
        io.generateOutputName(make_args(encode_method="prores"), "C:/v/frames_%05d.png")
        == "frames.mov"
    )


def test_sequence_input_default_extension():
    assert io.generateOutputName(make_args(), "C:/v/frames_%05d.png") == "frames.mp4"


# ---- #9 unsafe characters scrubbed ---------------------------------------- #


def test_unsafe_chars_sanitised():
    name = io.generateOutputName(
        make_args(restore=1, restore_method="a/b:c*?"), "clip.mp4"
    )
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


def test_sequence_detection_rejects_missing_frames(tmp_path):
    (tmp_path / "frame_001.png").touch()
    (tmp_path / "frame_003.png").touch()

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
    p = io.generateOutputPath(
        "X/a.mp4", explicit, out, make_args(output=explicit), used
    )
    assert p == explicit


def test_explicit_output_file_disambiguated_in_batch(tmp_path):
    out = str(tmp_path)
    explicit = str(tmp_path / "final.mp4")
    used = set()
    p1 = io.generateOutputPath(
        "A/a.mp4", explicit, out, make_args(output=explicit), used
    )
    p2 = io.generateOutputPath(
        "B/b.mp4", explicit, out, make_args(output=explicit), used
    )
    assert os.path.basename(p1) == "final.mp4"
    assert os.path.basename(p2) == "final-1.mp4"


# ---- #11 png sequence output -> unique folder ------------------------------ #


def test_png_sequence_creates_folder(tmp_path):
    out = str(tmp_path)
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4", None, out, make_args(encode_method="png"), used
    )
    assert p.endswith(os.path.join("clip", "frames_%05d.png"))
    assert os.path.isdir(os.path.dirname(p))


def test_png_sequence_folder_collision_bumped(tmp_path):
    out = str(tmp_path)
    used = set()
    io.generateOutputPath("X/clip.mp4", None, out, make_args(encode_method="png"), used)
    p2 = io.generateOutputPath(
        "Y/clip.mp4", None, out, make_args(encode_method="png"), used
    )
    assert "clip-1" in p2


def test_png_sequence_with_explicit_output_file_contained(tmp_path):
    # --encode_method png --output final.mp4 must not spray frames into the
    # parent directory: the explicit file's stem becomes the sequence folder.
    explicit = str(tmp_path / "final.mp4")
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4", explicit, str(tmp_path), make_args(encode_method="png"), used
    )
    assert p == os.path.join(str(tmp_path), "final", "frames_%05d.png")
    assert os.path.isdir(os.path.dirname(p))
    assert "%" in os.path.basename(p)  # runOutcome must see a sequence


def test_png_sequence_explicit_pattern_honoured(tmp_path):
    # An explicit printf-style pattern is already a valid sequence target.
    explicit = str(tmp_path / "frames_%05d.png")
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4", explicit, str(tmp_path), make_args(encode_method="png"), used
    )
    assert p == explicit


def test_autoclip_resolves_txt_extension():
    # autoclip writes a cut list; a fixed install-dir autoclipresults.txt
    # clobbered itself across a batch and ignored --output entirely.
    assert (
        io.generateOutputName(make_args(autoclip=1), "clip.mp4") == "clip-Autoclip.txt"
    )


def test_autoclip_batch_gets_unique_txt_paths(tmp_path):
    out = str(tmp_path)
    used = set()
    p1 = io.generateOutputPath("A/clip.mp4", None, out, make_args(autoclip=1), used)
    p2 = io.generateOutputPath("B/clip.mp4", None, out, make_args(autoclip=1), used)
    assert p1.endswith(".txt") and p2.endswith(".txt")
    assert p1 != p2


def test_explicit_txt_output_honoured(tmp_path):
    explicit = str(tmp_path / "cuts.txt")
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4", explicit, str(tmp_path), make_args(autoclip=1), used
    )
    assert p == explicit


def test_gif_encode_resolves_gif_extension():
    # gif into the input's container fails the muxer with a broken pipe.
    assert (
        io.generateOutputName(make_args(encode_method="gif", resize=1), "clip.mp4")
        == "clip-Resize2.gif"
    )


def test_jpeg_sequence_creates_folder(tmp_path):
    out = str(tmp_path)
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4", None, out, make_args(encode_method="jpeg"), used
    )
    assert p.endswith(os.path.join("clip", "frames_%05d.jpg"))
    assert os.path.isdir(os.path.dirname(p))


def test_jpeg_sequence_with_explicit_output_file_contained(tmp_path):
    explicit = str(tmp_path / "final.mp4")
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4", explicit, str(tmp_path), make_args(encode_method="jpeg"), used
    )
    assert p == os.path.join(str(tmp_path), "final", "frames_%05d.jpg")
    assert os.path.isdir(os.path.dirname(p))
    assert "%" in os.path.basename(p)  # runOutcome must see a sequence


def test_jpeg_encode_drops_extension_for_sequence_dir():
    assert io.generateOutputName(make_args(encode_method="jpeg"), "clip.mp4") == "clip"


def test_png_passthrough_stays_a_file_not_folder(tmp_path):
    out = str(tmp_path)
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4",
        None,
        out,
        make_args(encode_method="png", png_passthrough=1, single_image_input=1),
        used,
    )
    assert p.endswith(".png")
    assert "frames_%05d" not in p


def test_jpeg_single_image_stays_a_file_not_folder(tmp_path):
    out = str(tmp_path)
    used = set()
    p = io.generateOutputPath(
        "X/clip.mp4",
        None,
        out,
        make_args(encode_method="jpeg", png_passthrough=1, single_image_input=1),
        used,
    )
    assert p.endswith(".jpg")
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
    a = tmp_path / "a.mp4"
    a.touch()
    b = tmp_path / "b.mp4"
    b.touch()
    files = io.getVideoFiles(f"{a};{b}")
    assert sorted(os.path.basename(f) for f in files) == ["a.mp4", "b.mp4"]


def test_url_passed_through():
    assert io.getVideoFiles("https://youtu.be/x") == ["https://youtu.be/x"]


def test_sequence_pattern_passed_through():
    assert io.getVideoFiles("C:/v/frames_%05d.png") == ["C:/v/frames_%05d.png"]


def test_process_returns_sorted_unique_list(tmp_path):
    vdir = tmp_path / "vids"
    vdir.mkdir()
    out = tmp_path / "out"
    for nm in ["c.mp4", "a.mp4", "b.mp4"]:
        (vdir / nm).touch()
    res = io.processInputOutputPaths(make_args(input=str(vdir)), str(out))
    assert isinstance(res, list) and len(res) == 3
    assert [os.path.basename(r["videoPath"]) for r in res] == [
        "a.mp4",
        "b.mp4",
        "c.mp4",
    ]
    assert len({r["outputPath"] for r in res}) == 3  # all unique


def test_process_txt_list_input(tmp_path):
    vdir = tmp_path / "vids"
    vdir.mkdir()
    a = vdir / "a.mp4"
    a.touch()
    b = vdir / "b.mp4"
    b.touch()
    lst = tmp_path / "list.txt"
    lst.write_text(f'"{a}"\n{b}\n')
    res = io.processInputOutputPaths(make_args(input=str(lst)), str(tmp_path / "out"))
    assert len(res) == 2


def test_process_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.processInputOutputPaths(
            make_args(input=str(tmp_path / "nope.mp4")), str(tmp_path / "out")
        )


# --------------------------------------------------------------------------- #
# #13 validateEncoder webm handling
# --------------------------------------------------------------------------- #


def test_webm_without_compatible_encoder_falls_back_to_vp9():
    assert io.validateEncoder("x.webm", "h264", None) == "vp9"


def test_webm_with_custom_encoder_kept():
    assert io.validateEncoder("x.webm", "h264", "-c:v libx264") == "h264"


def test_webm_with_compatible_encoder_kept():
    assert io.validateEncoder("x.webm", "nvenc_av1", None) == "nvenc_av1"


@pytest.mark.parametrize(
    "method", ["vp9_nelux", "av1_nelux", "slow_av1_nelux", "nvenc_av1_nelux"]
)
def test_webm_keeps_compatible_nelux_encoders(method):
    # av1_nelux/nvenc_av1_nelux used to be force-swapped to FFmpeg's vp9 on a
    # .webm output, silently costing the run its in-process writer.
    assert io.validateEncoder("x.webm", method, None) == method


def test_webm_incompatible_nelux_encoder_stays_on_the_nelux_writer():
    assert io.validateEncoder("x.webm", "x264_nelux", None) == "vp9_nelux"


def test_non_webm_untouched():
    assert io.validateEncoder("x.mp4", "h264", None) == "h264"


# --------------------------------------------------------------------------- #
# EXTENSIONS vs INPUT_EXTENSIONS: one list used to serve both "what does a
# folder batch pick up" and "is --output a file or a folder". It omitted .m4v,
# so a folder mixing .m4v with .mp4 produced fewer outputs than inputs at exit
# 0, and `--output clip.m4v` silently made a DIRECTORY named clip.m4v.
# --------------------------------------------------------------------------- #


def test_directory_batch_picks_up_m4v_next_to_mp4(tmp_path):
    # The regression: .m4v is advertised in PARAMETERS.MD but was skipped in
    # silence, so the batch ended successfully having processed half the folder.
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "clip.m4v").touch()
    files = io.getVideoFiles(str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["clip.m4v", "clip.mp4"]


def test_directory_batch_picks_up_uppercase_m4v(tmp_path):
    # The extension test lowercases, so an .M4V from a camera/muxer counts too.
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "clip.M4V").touch()
    files = io.getVideoFiles(str(tmp_path))
    assert sorted(os.path.basename(f) for f in files) == ["clip.M4V", "clip.mp4"]


def test_directory_batch_names_skipped_ts_in_a_warning(tmp_path, capsys):
    # .ts is deliberately not scanned (TypeScript at least as often as MPEG-TS),
    # but a skip nobody can see is the bug this fix is about: say it out loud.
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "clip.m4v").touch()
    (tmp_path / "stream.ts").touch()

    files = io.getVideoFiles(str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["clip.m4v", "clip.mp4"]

    out = capsys.readouterr().out
    assert "stream.ts" in out
    assert "--input" in out  # tells the user how to process it anyway


def test_directory_batch_silent_about_ordinary_clutter(tmp_path, capsys):
    # A warning that fires on every normal media folder is noise users learn to
    # ignore, which would cost the .ts warning above all of its value.
    (tmp_path / "clip.mp4").touch()
    for clutter in ("notes.srt", "poster.jpg", "Thumbs.db", "desktop.ini"):
        (tmp_path / clutter).touch()

    files = io.getVideoFiles(str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["clip.mp4"]
    assert capsys.readouterr().out == ""


def test_explicit_m4v_output_stays_a_file(tmp_path):
    # `--output clip.m4v` used to create a DIRECTORY named clip.m4v holding a
    # differently-named .mp4, so every wrapper script lost track of its output.
    src = tmp_path / "in.mp4"
    src.touch()
    explicit = str(tmp_path / "out.m4v")
    res = io.processInputOutputPaths(
        make_args(input=str(src), output=explicit), str(tmp_path / "default")
    )
    assert [r["outputPath"] for r in res] == [explicit]
    assert not os.path.isdir(explicit)


def test_wmv_output_becomes_a_directory_but_warns(tmp_path, capsys):
    # TAS cannot mux into ASF, so the folder fallback stands -- but silently
    # renaming the user's file to a folder is what broke wrapper scripts.
    src = tmp_path / "in.mp4"
    src.touch()
    explicit = str(tmp_path / "clip.wmv")
    res = io.processInputOutputPaths(
        make_args(input=str(src), output=explicit), str(tmp_path / "default")
    )
    assert os.path.isdir(explicit)
    assert os.path.dirname(res[0]["outputPath"]) == explicit

    out = capsys.readouterr().out
    assert "clip.wmv" in out
    assert "FOLDER" in out and "not a file" in out  # both interpretations named
    assert ".mp4" in out  # and what it could have written instead


def test_plain_output_folder_creates_dir_without_warning(tmp_path, capsys):
    # The most common invocation there is; a false positive here would be worse
    # than the bug the warning exists for.
    src = tmp_path / "in.mp4"
    src.touch()
    explicit = str(tmp_path / "renders")  # does not exist yet
    res = io.processInputOutputPaths(
        make_args(input=str(src), output=explicit), str(tmp_path / "default")
    )
    assert os.path.isdir(explicit)
    assert os.path.dirname(res[0]["outputPath"]) == explicit
    assert capsys.readouterr().out == ""


def test_output_folder_with_a_dot_in_its_name_does_not_warn(tmp_path, capsys):
    # `ep01.v2` is a directory that merely has a dot in it, not a container.
    src = tmp_path / "in.mp4"
    src.touch()
    explicit = str(tmp_path / "ep01.v2")
    io.processInputOutputPaths(
        make_args(input=str(src), output=explicit), str(tmp_path / "default")
    )
    assert os.path.isdir(explicit)
    assert capsys.readouterr().out == ""


def test_mkv_input_extension_is_copied_but_wmv_falls_back_to_mp4():
    # The copied extension is what selects the muxer, and the default encoder
    # cannot mux into ASF: widening the batch scan to read-only containers would
    # otherwise turn a silent skip into a header-write crash after model load.
    assert io.generateOutputName(make_args(), "C:/v/clip.mkv") == "clip.mkv"
    assert io.generateOutputName(make_args(), "C:/v/clip.wmv") == "clip.mp4"


def _parametersMdFormats():
    """Parse the two video lines of PARAMETERS.MD's Supported Input Formats."""
    import re

    docPath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PARAMETERS.MD"
    )
    with open(docPath, encoding="utf-8") as handle:
        text = handle.read()

    block = re.search(r"#### Supported Input Formats\s*```(.*?)```", text, re.S)
    assert block, "PARAMETERS.MD lost its Supported Input Formats block"

    def parseLine(label):
        line = re.search(rf"{label}:(.*)", block.group(1))
        assert line, f"PARAMETERS.MD lost its '{label}' line"
        return {
            part.strip().lower()
            for part in line.group(1).split(",")
            if part.strip().startswith(".")
        }

    return parseLine(r"Video \(written and read\)"), parseLine(r"Video \(read only\)")


def test_parameters_md_matches_the_two_extension_lists():
    # Doc parity in both directions: the doc advertised .m4v that the code did
    # not scan, which is exactly how the batch drop went unnoticed.
    written, readOnly = _parametersMdFormats()
    assert written == set(io.EXTENSIONS)
    assert written | readOnly == set(io.INPUT_EXTENSIONS)
    assert readOnly.isdisjoint(io.EXTENSIONS)  # read-only means TAS won't write it
    assert ".m4v" in written and ".m4v" in io.EXTENSIONS
