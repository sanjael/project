"""
config.py - Centralised application settings loaded from .env
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CONFIDENCE_THRESHOLD: float = 0.75
    MODEL_DETECTION_PATH: str = "models/detection_resnet101.pth"
    MODEL_CLASSIFICATION_PATH: str = "models/classification_resnet101.pth"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
