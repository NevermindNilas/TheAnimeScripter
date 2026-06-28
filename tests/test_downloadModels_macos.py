from src.model import downloadModels as dm


def test_mps_download_reuses_base_weight_folder(tmp_path, monkeypatch):
    calls = []

    def fake_download(model, filename, download_url, folder_path, **_kwargs):
        calls.append((model, filename, download_url, folder_path))
        return str(tmp_path / "downloaded.pth")

    monkeypatch.setattr(dm, "weightsDir", str(tmp_path))
    monkeypatch.setattr(dm, "downloadAndLog", fake_download)

    dm.downloadModels("rife4.25-mps")

    assert calls == [
        (
            "rife4.25",
            "rife425.pth",
            f"{dm.TASURL}rife425.pth",
            str(tmp_path / "rife4.25"),
        )
    ]
