import json
import logging

import anthropic

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — ассистент службы поддержки. Классифицируешь обращения клиентов.

Верни ТОЛЬКО валидный JSON без каких-либо пояснений, точно в таком формате:
{
  "category": "продажи" | "поддержка" | "жалоба" | "спам",
  "priority": "low" | "medium" | "high",
  "summary": "суть обращения в одном коротком предложении на русском",
  "suggested_reply": "короткий вежливый черновик ответа клиенту на русском"
}
"""


class LLMClientError(Exception):
    """Базовая ошибка при обращении к LLM."""


class LLMUnavailableError(LLMClientError):
    """LLM API недоступен (сеть, таймаут, 5xx)."""


class LLMBadResponseError(LLMClientError):
    """LLM ответил, но не валидным JSON нужного формата."""


def _extract_text(message: anthropic.types.Message) -> str:
    parts = [block.text for block in message.content if block.type == "text"]
    return "".join(parts).strip()


def _parse_json(raw_text: str) -> dict:
    # LLM иногда оборачивает JSON в ```json ... ``` — на всякий случай чистим.
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        cleaned = cleaned.replace("json\n", "", 1)
    return json.loads(cleaned)


def classify_text(text: str, retries: int = 1) -> dict:
    """
    Отправляет текст в LLM и возвращает распарсенный JSON-ответ.
    Делает до `retries` повторных попыток, если ответ пришёл не в JSON.
    """
    if not settings.anthropic_api_key:
        raise LLMUnavailableError("ANTHROPIC_API_KEY не задан в окружении")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            message = client.messages.create(
                model=settings.model,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
        except anthropic.APIConnectionError as e:
            raise LLMUnavailableError(f"Не удалось связаться с LLM API: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMUnavailableError(f"Превышен лимит запросов к LLM API: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMUnavailableError(f"LLM API вернул ошибку {e.status_code}: {e.message}") from e

        raw_text = _extract_text(message)
        try:
            return _parse_json(raw_text)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning("Попытка %s: LLM вернул не-JSON ответ: %r", attempt + 1, raw_text)
            continue

    raise LLMBadResponseError(f"LLM не вернул валидный JSON после {retries + 1} попыток") from last_error
