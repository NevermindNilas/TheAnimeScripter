"""Regression tests for src.model.download download-failure handling.

Covers the truncated-download path: when the server advertises a content-length
the read loop never fully receives, downloadAndLog must surface a clean,
stringifiable, *retryable* exception. The previous code raised
``IncompleteRead(downloadedBytes, ...)`` with INTEGER args; its __repr__/__str__
does ``len(self.partial)``, so the very next ``logging.error(f"...{e}")`` raised
``TypeError: object of type 'int' has no len()`` -- a type NOT in the except
tuple -- which escaped and aborted the retry loop instead of retrying.
"""

import contextlib
import os

import pytest

# downloadAndLog lazily imports progressBarLogic -> barflow, and these tests
# patch ProgressBarDownloadLogic inside that module, so skip cleanly in the
# minimal CI env that does not install the full runtime deps (mirrors the
# torch/nelux importorskip in the other test modules).
pytest.importorskip("barflow")

import src.model.download as dm  # noqa: E402


class _TruncatedResponse:
    """Advertises 100 bytes via content-length but only yields 40."""

    def __init__(self):
        self.headers = {"content-length": "100"}
        self._chunks = [b"x" * 40, b""]
        self._i = 0

    def getcode(self):
        return 200

    def read(self, _n=-1):
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


@contextlib.contextmanager
def _noopProgressBar(_total, title=""):
    yield lambda _n: None


def testTruncatedDownloadRaisesStringifiableRetryable(tmp_path, monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda _url: _TruncatedResponse())
    monkeypatch.setattr(
        "src.infra.progressBarLogic.ProgressBarDownloadLogic", _noopProgressBar
    )

    with pytest.raises(Exception) as excInfo:
        dm.downloadAndLog(
            model="flownets",
            filename="dummy.pth",
            download_url="http://example.invalid/dummy.pth",
            folderPath=str(tmp_path),
            retries=1,
        )

    exc = excInfo.value
    # The defining symptom of the bug: stringifying the raised exception blew up
    # with a TypeError. It must now be a normal, retryable exception.
    assert not isinstance(exc, TypeError)
    assert isinstance(exc, ConnectionError)
    # The except handler does f"...{e}"; stringifying must not raise.
    assert "incomplete" in str(exc).lower()


def testTruncatedDownloadDoesNotCommitPartialFile(tmp_path, monkeypatch):
    # The truncated temp file must not be left behind / renamed into the cache.
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda _url: _TruncatedResponse())
    monkeypatch.setattr(
        "src.infra.progressBarLogic.ProgressBarDownloadLogic", _noopProgressBar
    )

    with pytest.raises(ConnectionError):
        dm.downloadAndLog(
            model="flownets",
            filename="dummy.pth",
            download_url="http://example.invalid/dummy.pth",
            folderPath=str(tmp_path),
            retries=1,
        )

    assert not (tmp_path / "dummy.pth").exists()
    assert not (tmp_path / "TEMP" / "dummy.pth").exists()


# --- resolveWeightDir: multi-file weight folders ---------------------------
#
# downloadModels creates weights/<model>/ *before* fetching a byte and extracts
# the zip straight into it, so a Ctrl-C / dropped connection / full disk leaves
# a folder that exists but holds nothing the loader needs (often just TEMP/).
# get_gmfss_model_dir used to guard on that folder merely existing, so every
# later run took the "already have it" branch and never retried -- GMFSS stayed
# broken with "rife.pkl not found" until the user deleted weights/gmfss by hand.
# resolveWeightDir guards on the members the loader actually opens instead.

_GMFSS_MEMBERS = (
    "rife.pkl",
    "flownet.pkl",
    "metric_union.pkl",
    "feat_union.pkl",
    "fusionnet_union.pkl",
)


def _stubWeightsDirAndDownload(tmp_path, monkeypatch, extracts=_GMFSS_MEMBERS):
    """Point resolveWeightDir at tmp_path; record + fake downloadModels.

    The fake mirrors the real one's shape: it creates weights/<model>/, drops
    the extracted members in it, and returns the path of a FILE inside that
    folder -- never the folder itself. Returns the list of models requested.
    """
    calls = []
    monkeypatch.setattr(dm, "weightsDir", str(tmp_path))

    def fakeDownloadModels(model=None, **_kwargs):
        calls.append(model)
        modelDir = tmp_path / model
        modelDir.mkdir(parents=True, exist_ok=True)
        for member in extracts:
            (modelDir / member).write_bytes(b"weights")
        return str(modelDir / f"{model}.zip")

    monkeypatch.setattr(dm, "downloadModels", fakeDownloadModels)
    return calls


def testResolveWeightDirRedownloadsFolderHoldingOnlyTemp(tmp_path, monkeypatch):
    # The exact wreckage an interrupted download leaves: the folder exists and
    # contains only the TEMP/ staging dir. The folder-existence guard called
    # this "already downloaded" forever.
    calls = _stubWeightsDirAndDownload(tmp_path, monkeypatch)
    (tmp_path / "gmfss" / "TEMP").mkdir(parents=True)

    result = dm.resolveWeightDir("gmfss", _GMFSS_MEMBERS)

    assert calls == ["gmfss"]
    assert result == str(tmp_path / "gmfss")


def testResolveWeightDirRedownloadsPartialExtraction(tmp_path, monkeypatch):
    # A zip that stopped extracting part-way: most members present, one absent.
    # Any single missing member is a load failure, so it must re-download.
    calls = _stubWeightsDirAndDownload(tmp_path, monkeypatch)
    modelDir = tmp_path / "gmfss"
    modelDir.mkdir()
    for member in _GMFSS_MEMBERS[:-1]:
        (modelDir / member).write_bytes(b"weights")

    dm.resolveWeightDir("gmfss", _GMFSS_MEMBERS)

    assert calls == ["gmfss"]
    assert (modelDir / _GMFSS_MEMBERS[-1]).exists()


def testResolveWeightDirCompleteFolderSkipsDownload(tmp_path, monkeypatch):
    # The hot path: every run after the first must not hit the network.
    calls = _stubWeightsDirAndDownload(tmp_path, monkeypatch)
    modelDir = tmp_path / "gmfss"
    modelDir.mkdir()
    for member in _GMFSS_MEMBERS:
        (modelDir / member).write_bytes(b"weights")

    result = dm.resolveWeightDir("gmfss", _GMFSS_MEMBERS)

    assert calls == []
    assert result == str(modelDir)


def testResolveWeightDirDownloadsWhenFolderAbsent(tmp_path, monkeypatch):
    # Plain first run: nothing on disk at all.
    calls = _stubWeightsDirAndDownload(tmp_path, monkeypatch)

    result = dm.resolveWeightDir("gmfss", _GMFSS_MEMBERS)

    assert calls == ["gmfss"]
    assert os.path.isdir(result)
    for member in _GMFSS_MEMBERS:
        assert os.path.exists(os.path.join(result, member))


def testResolveWeightDirReturnsModelDirNotArchive(tmp_path, monkeypatch):
    # downloadModels returns a path to a file *inside* the folder (the archive
    # or an extracted member). The loader is handed a directory and joins the
    # .pkl names onto it, so resolveWeightDir must hand back the folder.
    _stubWeightsDirAndDownload(tmp_path, monkeypatch)

    result = dm.resolveWeightDir("gmfss", _GMFSS_MEMBERS)

    assert not result.endswith(".zip")
    assert os.path.isdir(result)
    assert result == str(tmp_path / "gmfss")


def testGmfssRequiredMembersMatchWhatTheLoaderOpens():
    # GMFSS_REQUIRED_MEMBERS is hardcoded, while the torch.load calls it must
    # mirror live in src/gmfss/, a vendored tree ruff does not lint -- nothing
    # else would notice the two drifting apart. Parse the loader instead of
    # trusting it. (src.gmfss.gmfss imports torch and builds a CudaChecker at
    # module scope, hence the guard.)
    pytest.importorskip("torch")
    import ast
    from pathlib import Path

    from src.gmfss.gmfss import GMFSS_REQUIRED_MEMBERS

    loaderPath = (
        Path(__file__).resolve().parents[1] / "src" / "gmfss" / "model" / "GMFSS.py"
    )
    tree = ast.parse(loaderPath.read_text(encoding="utf-8"))

    loaded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            continue
        for sub in ast.walk(node.args[0]):
            if isinstance(sub, ast.Constant) and str(sub.value).endswith(".pkl"):
                loaded.add(sub.value)

    assert loaded, f"found no torch.load(*.pkl) calls in {loaderPath}"
    assert set(GMFSS_REQUIRED_MEMBERS) == loaded


# --- downloadAndLog: 404 handling ------------------------------------------
#
# A 404 means the asset was never published (a CLI choice whose weight is
# missing from the release), not a flaky network: retrying it twice more only
# delays an identical failure, and the bare "HTTP Error 404" named neither the
# model nor the file, so nobody could tell which choice to stop using.


class _HttpErrorOpener:
    """urlopen replacement that always fails with `code`, counting attempts."""

    def __init__(self, code):
        self.code = code
        self.calls = 0

    def __call__(self, url):
        from urllib.error import HTTPError

        self.calls += 1
        raise HTTPError(url, self.code, "Not Found", None, None)


def testNotFoundIsNotRetriedAndNamesModelAndFile(tmp_path, monkeypatch):
    import urllib.request
    from urllib.error import HTTPError

    opener = _HttpErrorOpener(404)
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    messages = []
    monkeypatch.setattr(dm, "logAndPrint", lambda msg, *a, **k: messages.append(msg))

    with pytest.raises(HTTPError):
        dm.downloadAndLog(
            model="flownets",
            filename="dummy.pth",
            download_url="http://example.invalid/dummy.pth",
            folderPath=str(tmp_path),
            retries=3,
        )

    assert opener.calls == 1, "a 404 must fail fast, not burn all three attempts"
    joined = " ".join(messages)
    assert "flownets" in joined
    assert "dummy.pth" in joined


def testServerErrorIsStillRetried(tmp_path, monkeypatch):
    # The fail-fast branch must stay 404-only: a 5xx is exactly the flaky-server
    # case the retry loop exists for.
    import urllib.request
    from urllib.error import HTTPError

    opener = _HttpErrorOpener(500)
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    monkeypatch.setattr(dm, "logAndPrint", lambda *a, **k: None)

    with pytest.raises(HTTPError):
        dm.downloadAndLog(
            model="flownets",
            filename="dummy.pth",
            download_url="http://example.invalid/dummy.pth",
            folderPath=str(tmp_path),
            retries=3,
        )

    assert opener.calls == 3
