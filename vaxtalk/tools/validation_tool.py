"""
Safety Validation Tool

Provides the `get_draft_for_validation` FunctionTool that retrieves the draft
response from session state for the SafetyCheckAgent to validate.

The actual validation logic is defined in the SafetyCheckAgent's instruction
(SAFETY_CHECK_INSTRUCTION in prompts.py). This tool simply provides access
to the draft_response and rag_output stored in session state.
"""

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from vaxtalk.config.logging_config import get_logger

logger = get_logger(__name__)


######################################
## VALIDATION TOOL
######################################


@FunctionTool
def get_draft_for_validation(tool_context: ToolContext) -> dict[str, str]:
    """
    Retrieve the draft response and RAG output from session state.

    Call this tool FIRST to get the content that needs safety validation.
    After receiving the output, analyze the draft_response for safety issues
    following your instructions.

    Args:
        tool_context: The ADK tool context for accessing session state

    Returns:
        Dict containing:
        - status: "ready" if draft exists, "error" if missing
        - draft_response: The text to validate (from DraftComposerAgent)
        - rag_output: Original RAG sources for verifying citations

    Examples:
        >>> result = get_draft_for_validation(tool_context)
        >>> if result["status"] == "ready":
        ...     # Analyze draft_response for safety violations
        ...     pass
    """
    draft_response = tool_context.state.get("draft_response", "")
    rag_output = tool_context.state.get("rag_output", "")

    if not draft_response:
        logger.warning("No draft_response found in session state")
        return {
            "status": "error",
            "error_message": "No draft response found. The DraftComposerAgent may have failed.",
            "draft_response": "",
            "rag_output": rag_output,
        }

    logger.info("Retrieved draft response for validation (%d characters)", len(draft_response))

    return {
        "status": "ready",
        "draft_response": draft_response,
        "rag_output": rag_output,
    }


logger.info("Safety validation tool configured.")
