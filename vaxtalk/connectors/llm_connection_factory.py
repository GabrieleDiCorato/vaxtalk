from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.genai.types import HttpRetryOptions
from vaxtalk.config.logging_config import get_logger

logger = get_logger(__name__)


class LlmConnectionFactory:
    """Simple internal factory to get LLM connections based on type.
    We only support connections that have been tested and verified within VaxTalk.
    """

    @staticmethod
    def get_llm_connection(
        model_full_name: str,
        retry_config: HttpRetryOptions,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> BaseLlm:
        """Get an LLM connection based on model name.

        Args:
            model_full_name: Full name of the model (e.g., "gemini-2.5-flash", "openrouter/deepseek/deepseek-r1")
            retry_config: Retry configuration for API calls
            api_key: API key for authentication
            api_base: Optional API base URL for custom endpoints
        Returns:
            An instance of BaseLlm for the specified model.
        """

        if model_full_name.startswith("gemini"):
            logger.info("Creating Gemini LLM connection for model: %s", model_full_name)
            return Gemini(
                model=model_full_name,
                retry_options=retry_config
            )
        elif model_full_name.startswith("openrouter"):
            logger.info("Creating LiteLlm connection for OpenRouter model: %s", model_full_name)
            return LiteLlm(
                model=model_full_name,
                api_key=api_key,
                api_base=api_base
            )
        else:
            logger.error("Unsupported model requested: %s", model_full_name)
            raise ValueError(f"Unsupported model: {model_full_name}")
