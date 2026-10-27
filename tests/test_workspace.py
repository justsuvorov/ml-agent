from pathlib import Path

from hamcrest import assert_that, equal_to, raises

from ml_agent.workspace import Folder, MissingFileError, OutsideFileError
from tests.titanic import name


def test_finds_a_file_inside_the_folder(tmp_path: Path) -> None:
    file = name()
    (tmp_path / file).write_text(name(), encoding="utf-8")
    assert_that(
        Folder(tmp_path).file(file),
        equal_to((tmp_path / file).resolve()),
        "Folder did not find the file inside it",
    )


def test_refuses_a_file_outside_the_folder(tmp_path: Path) -> None:
    file = name()
    (tmp_path / file).write_text(name(), encoding="utf-8")
    (tmp_path / "inner").mkdir()
    assert_that(
        lambda: Folder(tmp_path / "inner").file(f"../{file}"),
        raises(OutsideFileError),
        "Folder did not refuse the file outside it",
    )


def test_refuses_a_missing_file(tmp_path: Path) -> None:
    assert_that(
        lambda: Folder(tmp_path).file(name()),
        raises(MissingFileError),
        "Folder did not refuse the missing file",
    )
