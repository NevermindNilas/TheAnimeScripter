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
