from pathlib import Path
from typing import Final, Protocol, final, override

from ml_agent.tool import ToolError


class Workspace(Protocol):
    """Place where the agent keeps configs, data and results of a session."""

    def file(self, name: str) -> Path: ...

    def place(self, name: str) -> Path: ...


@final
class Folder(Workspace):
    """Directory on disk that serves as the workspace and never lets a file name escape it."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root: Final = root

    @override
    def file(self, name: str) -> Path:
        if not (self._root / name).resolve().is_relative_to(self._root.resolve()):
            raise OutsideFileError(name)
        if not (self._root / name).is_file():
            raise MissingFileError(name)
        return (self._root / name).resolve()

    @override
    def place(self, name: str) -> Path:
        if not (self._root / name).resolve().is_relative_to(self._root.resolve()):
            raise OutsideFileError(name)
        (self._root / name).resolve().parent.mkdir(parents=True, exist_ok=True)
        return (self._root / name).resolve()


@final
class OutsideFileError(ToolError):
    """File name that points outside of the workspace."""

    def __init__(self, name: str) -> None:
        super().__init__(f"File {name} lies outside of the workspace")


@final
class MissingFileError(ToolError):
    """File name that points to nothing in the workspace."""

    def __init__(self, name: str) -> None:
        super().__init__(f"File {name} does not exist in the workspace")
