from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="Текст обращения клиента для классификации",
    )


class ClassifyResponse(BaseModel):
    category: str = Field(..., description="Категория обращения")
    priority: Priority = Field(..., description="Приоритет обработки")
    summary: str = Field(..., description="Суть обращения в одном предложении")
    suggested_reply: str = Field(..., description="Черновик ответа клиенту")


class HealthResponse(BaseModel):
    status: str = "ok"
