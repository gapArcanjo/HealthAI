import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("ENV", "development")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "300"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    emergency_phone: str = os.getenv("EMERGENCY_PHONE", "192")
    allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ) or ("*",)
    trusted_hosts: tuple[str, ...] = tuple(
        host.strip()
        for host in os.getenv("TRUSTED_HOSTS", "*").split(",")
        if host.strip()
    ) or ("*",)


settings = Settings()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
