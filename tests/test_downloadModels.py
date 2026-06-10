"""Regression tests for src/utils/downloadModels.py download-failure handling.

Covers the truncated-download path: when the server advertises a content-length
the read loop never fully receives, downloadAndLog must surface a clean,
stringifiable, *retryable* exception. The previous code raised
``IncompleteRead(downloadedBytes, ...)`` with INTEGER args; its __repr__/__str__
does ``len(self.partial)``, so the very next ``logging.error(f"...{e}")`` raised
``TypeError: object of type 'int' has no len()`` -- a type NOT in the except
tuple -- which escaped and aborted the retry loop instead of retrying.
"""

import contextlib

import pytest

# downloadAndLog lazily imports progressBarLogic -> barflow, and these tests
# patch ProgressBarDownloadLogic inside that module, so skip cleanly in the
# minimal CI env that does not install the full runtime deps (mirrors the
# torch/nelux importorskip in the other test modules).
pytest.importorskip("barflow")

from src.utils import downloadModels as dm  # noqa: E402


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
        "src.utils.progressBarLogic.ProgressBarDownloadLogic", _noopProgressBar
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
        "src.utils.progressBarLogic.ProgressBarDownloadLogic", _noopProgressBar
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
