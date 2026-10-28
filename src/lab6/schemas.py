"""API contracts for Lab 6."""

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


class PredictResponse(BaseModel):
    version: str
    predicted_class: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class PromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: StrictStr = Field(min_length=1)
    reason: StrictStr = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def reject_blank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reason must not be blank")
        return value


class SwitchResponse(BaseModel):
    action: str
    previous_version: str
    active_version: str
    message: str
