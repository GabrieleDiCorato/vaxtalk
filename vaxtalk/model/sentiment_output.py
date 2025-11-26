from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class Intensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SentimentOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=True,
    )

    frustration: Intensity = Field(..., description="The level of frustration expressed by the user.")
    confusion: Intensity = Field(..., description="The level of confusion expressed by the user.")
    satisfaction: Intensity = Field(..., description="The level of satisfaction expressed by the user.")
