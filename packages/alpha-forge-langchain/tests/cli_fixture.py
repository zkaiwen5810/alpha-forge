"""Offline CLI entrypoint used only by subprocess tests."""

from unittest.mock import patch

from alpha_forge_langchain import AgentApplication
from alpha_forge_langchain.cli import main
from langchain_core.messages import AIMessage
from test_agent_application import ScriptedModel


class OfflineApplication(AgentApplication):
    def __init__(self, database):
        """Override AgentApplication.__init__ to inject an offline model for CLI tests."""
        super().__init__(
            database,
            model=ScriptedModel(responses=[AIMessage(content="offline answer")]),
        )


if __name__ == "__main__":
    with patch("alpha_forge_langchain.cli.AgentApplication", OfflineApplication):
        raise SystemExit(main())
