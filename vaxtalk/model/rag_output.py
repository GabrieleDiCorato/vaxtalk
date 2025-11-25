from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class SuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    response: str = Field(..., description="The successful result data.")


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_message: str = Field(..., description="A message describing the error that occurred.")


class RagOutput(BaseModel):
    result: SuccessResponse | ErrorResponse = Field(..., discriminator="status")
