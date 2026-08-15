"""``--json`` must reject values argparse would have rejected.

The JSON path assigns straight onto the namespace, so every ``choices=`` list
the parser declares was bypassed. The After Effects bridge sent
``depth_method: "small_v3-directml"`` -- a method with no CLI choice and no
backend class -- and the run only failed several seconds later, inside
``src/factories/standalone.py:depth()``, after decoding metadata.
"""

import argparse
import json

import pytest

from src.cli.config import CliConfig
from src.cli.parser import _buildParser


@pytest.fixture
def parser(tmp_path):
    return _buildParser(str(tmp_path))


def runJson(parser, tmp_path, config):
    configPath = tmp_path / "config.json"
    configPath.write_text(json.dumps(config), encoding="utf-8")
    args = argparse.Namespace(**{a.dest: a.default for a in parser._actions})
    args.json = str(configPath)
    return CliConfig.fromArgs(args, parser, ["--json", str(configPath)])


def test_rejectsUnknownMethod(parser, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        runJson(parser, tmp_path, {"depth_method": "small_v3-directml"})

    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "small_v3-directml" in output
    assert "--list_methods depth" in output


def test_rejectsUnknownListElement(parser, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        runJson(parser, tmp_path, {"restore_method": ["anime1080fixer", "nope"]})

    assert excinfo.value.code == 1


def test_shortChoiceListIsSpelledOut(parser, tmp_path, capsys):
    with pytest.raises(SystemExit):
        runJson(parser, tmp_path, {"bit_depth": "9bit"})

    assert "8bit" in capsys.readouterr().out


def test_acceptsValidValues(parser, tmp_path):
    config = runJson(
        parser,
        tmp_path,
        {
            "depth_method": "small_v2-directml",
            "restore_method": ["anime1080fixer"],
            "bit_depth": "16bit",
        },
    )

    assert config.args.depth_method == "small_v2-directml"
    assert config.args.restore_method == ["anime1080fixer"]
    assert config.args.depth is True


def test_valueWithoutChoicesIsUntouched(parser, tmp_path):
    config = runJson(parser, tmp_path, {"input": "anything.mp4", "half": False})

    assert config.args.input == "anything.mp4"
    # `--half` declares `type=str2bool`, a plain function. Anything that reaches
    # for `isinstance(value, action.type)` raises TypeError here -- and because
    # presets store all of vars(args), that would break every preset load too.
    assert config.args.half is False


# --------------------------------------------------------------------------- #
# Type coercion. The choices check only ever guarded `choices=`; nothing ran
# `action.type`, so a panel or hand-written file that quoted a number carried a
# str into the run and died at main.py's `fps * interpolateFactor` with
# "can't multiply sequence by non-int of type 'float'" -- naming neither the
# option nor the file. The store_true half was worse: it failed silently.
# --------------------------------------------------------------------------- #


def test_coercesQuotedNumber(parser, tmp_path):
    config = runJson(parser, tmp_path, {"interpolate": True, "interpolate_factor": "2"})

    assert config.args.interpolate_factor == 2.0
    assert isinstance(config.args.interpolate_factor, float)


def test_coercesQuotedNumberThatIsAlsoAChoice(parser, tmp_path):
    # Coercion runs before the choices check, so this no longer dies on the
    # baffling `Invalid value ... '2'. Valid choices: 2, 3, 4.`
    config = runJson(parser, tmp_path, {"upscale": True, "upscale_factor": "2"})

    assert config.args.upscale_factor == 2


def test_rejectsUnparseableNumber(parser, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        runJson(parser, tmp_path, {"interpolate": True, "interpolate_factor": "two"})

    assert excinfo.value.code == 1
    assert "interpolate_factor" in capsys.readouterr().out


@pytest.mark.parametrize(
    "given,expected",
    [
        ("false", False),
        ("true", True),
        ("no", False),
        ("yes", True),
        (0, False),
        (1, True),
    ],
)
def test_storeTrueFlagIsParsedNotTruthyTested(parser, tmp_path, given, expected):
    # The silent half: "false" is a non-empty string, so the old straight
    # assignment turned interpolation ON for a config that asked for it off.
    config = runJson(parser, tmp_path, {"interpolate": given})

    assert config.args.interpolate is expected


def test_rejectsAmbiguousBoolean(parser, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        runJson(parser, tmp_path, {"interpolate": "maybe"})

    assert excinfo.value.code == 1
    assert "interpolate" in capsys.readouterr().out


def test_unknownKeyWarnsOnConsoleAndDoesNotExit(parser, tmp_path, capsys):
    # A typo'd key used to change nothing and say nothing: the warning went to
    # TAS-Log.log only. It must stay a warning -- an AE panel that serializes an
    # extra field must not be made unrunnable.
    config = runJson(parser, tmp_path, {"input": "x.mp4", "interpolat_factor": 3})

    output = capsys.readouterr().out
    assert "interpolat_factor" in output
    assert "interpolate_factor" in output  # difflib suggestion
    assert config.args.input == "x.mp4"
