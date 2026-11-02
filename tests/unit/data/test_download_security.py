"""Security tests for archive extraction (Zip Slip / Tar Slip)."""

import tarfile
import zipfile

import pytest

from flashdet.data.download import _extract


def test_zip_slip_is_rejected(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
        zf.writestr("safe.txt", "ok")

    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="unsafe archive member"):
        _extract(str(archive), str(dest))
    assert not (tmp_path / "escaped.txt").exists()


def test_absolute_zip_path_is_rejected(tmp_path):
    archive = tmp_path / "absolute.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/tmp/flashdet-absolute-escape.txt", "pwned")

    with pytest.raises(ValueError, match="unsafe archive member"):
        _extract(str(archive), str(tmp_path / "out"))


def test_tar_slip_is_rejected(tmp_path):
    archive = tmp_path / "evil.tar"
    payload = tmp_path / "payload.txt"
    payload.write_text("pwned")
    with tarfile.open(archive, "w") as tf:
        info = tf.gettarinfo(str(payload), arcname="../escaped.txt")
        with open(payload, "rb") as f:
            tf.addfile(info, f)

    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="unsafe archive member"):
        _extract(str(archive), str(dest))
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_symlink_escape_is_rejected(tmp_path):
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        tf.addfile(info)

    with pytest.raises(ValueError, match="archive link outside"):
        _extract(str(archive), str(tmp_path / "out"))


def test_safe_archive_extracts_normally(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/file.txt", "ok")

    dest = tmp_path / "out"
    _extract(str(archive), str(dest))
    assert (dest / "data" / "file.txt").read_text() == "ok"
