"""VaxTalk - lightweight package exports.

Importing :mod:`vaxtalk` should not spin up the full agent stack.  This module
exposes the commonly used ``root_agent``, ``vax_talk_assistant``, and ``runner``
attributes via lazy accessors so CLI entry points can load quickly while still
supporting ``from vaxtalk import root_agent`` for ADK hosting.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["root_agent", "vax_talk_assistant", "runner"]
_LAZY_ATTRS = set(__all__)


def __getattr__(name: str):
    """Lazily import heavy agent objects on demand."""

    if name in _LAZY_ATTRS:
        module = import_module("vaxtalk.agent")
        value = getattr(module, name)
        globals()[name] = value  # Cache for subsequent lookups
        return value
    raise AttributeError(f"module 'vaxtalk' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose lazy attributes to dir() callers for better discoverability."""

    return sorted(list(globals().keys()) + list(_LAZY_ATTRS))


if TYPE_CHECKING:  # pragma: no cover - assists type checkers without runtime cost
    from .agent import root_agent, vax_talk_assistant, runner
