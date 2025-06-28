"""Configuration settings for the application."""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    APP_NAME: str = "Automated Book Publication Workflow"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    SCREENSHOTS_DIR: Path = DATA_DIR / "screenshots"
    CONTENT_DIR: Path = DATA_DIR / "content"
    MODELS_DIR: Path = DATA_DIR / "models"
    
    SCRAPER_TIMEOUT: int = 30  
    SCRAPER_HEADLESS: bool = True
    
    OPENROUTER_API_KEY: Optional[str] = None
    
    DEFAULT_AI_MODEL: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    
    CHROMA_DB_PATH: Path = DATA_DIR / "chroma_db"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

settings = Settings()
settings.DATA_DIR.mkdir(exist_ok=True)
settings.SCREENSHOTS_DIR.mkdir(exist_ok=True)
settings.CONTENT_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)
