import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.llm_client import LLMBadResponseError, LLMUnavailableError, classify_text
from app.schemas import ClassifyRequest, ClassifyResponse, HealthResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LLM Request Classifier",
    description="Классифицирует обращения клиентов через LLM: категория, приоритет, черновик ответа.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    try:
        result = classify_text(request.text)
    except LLMUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"LLM сервис временно недоступен: {e}") from e
    except LLMBadResponseError as e:
        raise HTTPException(status_code=502, detail=f"LLM вернул неожиданный формат ответа: {e}") from e

    try:
        return ClassifyResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ответ LLM не соответствует ожидаемой структуре: {e}",
        ) from e
