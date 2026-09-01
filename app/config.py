import os


class Settings:
    """
    Настройки сервиса. Ключ API берётся из переменной окружения —
    никогда не хранится в коде.
    """

    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    model: str = os.environ.get("CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")


settings = Settings()
