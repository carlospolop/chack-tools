import subprocess

try:
    from agents import function_tool
except ImportError:
    function_tool = None

from .config import ToolsConfig
from .formatting import _truncate

class ExecTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def run(self, command: str) -> str:
        timeout = max(1, int(self.config.exec_timeout_seconds or 60))
        max_chars = max(1, int(self.config.exec_max_output_chars or 5000))
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=None,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip() or "(no output)"
        return _truncate(output, max_chars)


def get_exec_tool(helper: ExecTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="exec")
    def exec_tool(command: str) -> str:
        """Execute a shell command locally and return combined output.

        Use this to access local CLIs, curl/wget endpoints, and inspect files.
        If the output is large or truncated, re-run with grep/jq/sed to narrow it.
        Ideal for gathering evidence or checking system state; avoid destructive commands unless asked.

        Args:
            command: The shell command to execute.
        """
        return helper.run(command)

    return exec_tool
