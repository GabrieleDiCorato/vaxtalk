"""Model definitions for VaxTalk."""

from src.model.sentiment_output import Intensity, SentimentOutput
from src.model.rag_output import RagOutput, SuccessResponse, ErrorResponse

__all__ = ["Intensity", "SentimentOutput", "RagOutput", "SuccessResponse", "ErrorResponse"]
