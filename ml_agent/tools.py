from collections.abc import Mapping
from typing import Final, final, override

from pydantic import JsonValue

from ml_agent.tool import Tool, ToolError
from ml_agent.toolbox import Toolbox


@final
class Tools(Toolbox):
    """Tools of the agent, each known to the language model by its name."""

    __slots__ = ("_tools",)

    def __init__(self, tools: Mapping[str, Tool]) -> None:
        self._tools: Final = tools

    @override
    def manual(self) -> str:
        return "\n\n".join(f"### {name}\n{tool.manual()}" for name, tool in self._tools.items())

    @override
    def run(self, name: str, arguments: Mapping[str, JsonValue]) -> str:
        if name not in self._tools:
            raise UnknownToolError(name, tuple(self._tools))
        return self._tools[name].run(arguments)


@final
class UnknownToolError(ToolError):
    """Call of a tool that the agent does not have."""

    def __init__(self, name: str, names: tuple[str, ...]) -> None:
        super().__init__(f"Tool {name} does not exist, available tools are {', '.join(names)}")
