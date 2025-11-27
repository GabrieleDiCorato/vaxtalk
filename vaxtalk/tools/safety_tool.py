"""
Safety Review Tool

Provides the flag_for_human_review FunctionTool for flagging responses
that require human oversight due to safety concerns.
"""

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from vaxtalk.config.logging_config import get_logger

logger = get_logger(__name__)

######################################
## SAFETY TOOL
######################################


@FunctionTool
def flag_for_human_review(
    tool_context: ToolContext, reason: str, severity: str
) -> dict[str, str]:
    """
    Flag a response for human review when safety concerns are critical.

    Updates the session state with flagging information that can be used
    by downstream systems (notification queues, dashboards, etc.).

    Args:
        tool_context: The ADK tool context for state management
        reason: Explanation of why human review is needed
        severity: One of ["low", "medium", "high", "critical"]

    Returns:
        Dict containing status and severity level

    Examples:
        >>> result = flag_for_human_review(
        ...     tool_context,
        ...     reason="User asking about vaccine interactions with unlisted medication",
        ...     severity="high"
        ... )
        >>> print(result)
        {'status': 'flagged', 'severity': 'high'}
    """
    tool_context.state["flagged_for_review"] = True
    tool_context.state["flag_reason"] = reason
    tool_context.state["flag_severity"] = severity

    # In production, this could trigger a notification or queue system
    logger.warning("FLAGGED FOR REVIEW [%s]: %s", severity, reason)

    return {"status": "flagged", "severity": severity}


logger.info("Safety tool configured.")
