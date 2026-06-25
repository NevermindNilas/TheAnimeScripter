import logging

import src.constants as cs
import src.infra.checkSpecs as checkSpecs


def test_check_system_supports_darwin(monkeypatch, caplog):
    monkeypatch.setattr(cs, "SYSTEM", "Darwin", raising=False)
    monkeypatch.setattr(checkSpecs, "getMacosInfo", lambda: logging.info("mac ok"))

    with caplog.at_level(logging.INFO):
        checkSpecs.checkSystem()

    assert "mac ok" in caplog.text
    assert "Unsupported OS" not in caplog.text
