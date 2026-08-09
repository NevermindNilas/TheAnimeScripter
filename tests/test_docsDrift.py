"""Pin PARAMETERS.MD to parser.py.

The reference guide drifts silently: a flag gets added and never documented, or
gets renamed and the old name lingers in an example the user copy-pastes. Both
directions are checked here, at flag granularity only -- the model tables are
deliberately written as "base model + backend matrix" rather than spelling out
every `<model>-<backend>` choice, so a per-choice check would be pure noise.
"""

import re
from pathlib import Path

from src.cli.parser import _buildParser

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMETERS = REPO_ROOT / "PARAMETERS.MD"

# argparse builds these itself; they are not TAS options.
BUILTIN_FLAGS = {"--help", "--version"}


def _parserFlags():
    parser = _buildParser("output")
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    } - BUILTIN_FLAGS


# Long options of OTHER tools that may legitimately appear in examples
# (`pip install --upgrade`, ...). Without this the ghost check reads them as
# TAS flags the parser rejects. Never add a real TAS flag here -- the test
# below fails if one sneaks in.
FOREIGN_FLAGS = {
    "--upgrade",
    "--user",
    "--no-cache-dir",
    "--recursive",
}


def _documentedFlags():
    text = PARAMETERS.read_text(encoding="utf-8")
    # Table-of-contents anchors like `](#-output--options)` tokenize exactly like
    # a long flag; drop every link target before scanning.
    text = re.sub(r"\]\([^)]*\)", "]()", text)
    return set(re.findall(r"--[a-zA-Z][a-zA-Z0-9_]*", text))


def testForeignFlagAllowlistHidesNoRealFlag():
    collisions = sorted(FOREIGN_FLAGS & _parserFlags())
    assert not collisions, (
        f"FOREIGN_FLAGS would exempt real TAS flags from the drift check: {collisions}"
    )


def testEveryParserFlagIsDocumented():
    missing = sorted(_parserFlags() - _documentedFlags())
    assert not missing, (
        f"{len(missing)} CLI flags are missing from PARAMETERS.MD: {missing}"
    )


def testParametersDocumentsNoFlagThatDoesNotExist():
    ghosts = sorted(_documentedFlags() - _parserFlags() - BUILTIN_FLAGS - FOREIGN_FLAGS)
    assert not ghosts, (
        f"PARAMETERS.MD documents {len(ghosts)} flags the parser does not "
        f"accept: {ghosts}"
    )


def testDocumentedExamplesUseValidChoiceValues():
    """`--bit_depth 10bit` sat in two copy-pasteable examples; argparse rejects
    it (`8bit`/`16bit`), so following the guide exits 2."""
    parser = _buildParser("output")
    choicesByFlag = {
        option: {str(choice) for choice in action.choices}
        for action in parser._actions
        if action.choices
        for option in action.option_strings
    }
    text = PARAMETERS.read_text(encoding="utf-8")

    invalid = []
    for flag, value in re.findall(r"(--[a-zA-Z][a-zA-Z0-9_]*)\s+([^\s`\\|]+)", text):
        if flag not in choicesByFlag:
            continue
        value = value.strip("\"'")
        # Placeholders and prose, not literal values.
        if not value or value[0] in "<[{$" or value.endswith((",", ".", ":")):
            continue
        if value not in choicesByFlag[flag]:
            invalid.append(f"{flag} {value}")
    assert not invalid, (
        f"PARAMETERS.MD examples use values the parser rejects: {sorted(set(invalid))}"
    )


def testEveryEncodeMethodIsDocumented():
    """Unlike the model tables, the encoder tables list every method by name,
    so a new encoder that never reaches the docs is undiscoverable."""
    parser = _buildParser("output")
    (action,) = [a for a in parser._actions if "--encode_method" in a.option_strings]
    text = PARAMETERS.read_text(encoding="utf-8")
    missing = [
        choice
        for choice in action.choices
        if not re.search(rf"(?<![\w-]){re.escape(choice)}(?![\w-])", text)
    ]
    assert not missing, f"--encode_method choices missing from PARAMETERS.MD: {missing}"
