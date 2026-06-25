import logging
import sys
import types

import src.constants as cs


def test_check_system_supports_darwin(monkeypatch, caplog):
    fakePsutil = types.SimpleNamespace(virtual_memory=lambda: None)
    monkeypatch.setitem(sys.modules, "psutil", fakePsutil)

    import src.infra.checkSpecs as checkSpecs

    monkeypatch.setattr(cs, "SYSTEM", "Darwin", raising=False)
    monkeypatch.setattr(checkSpecs, "getMacosInfo", lambda: logging.info("mac ok"))

    with caplog.at_level(logging.INFO):
        checkSpecs.checkSystem()

    assert "mac ok" in caplog.text
    assert "Unsupported OS" not in caplog.text
