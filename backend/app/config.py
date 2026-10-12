from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./dev.db"
    max_upload_bytes: int = 5 * 1024 * 1024
    judge_model: str = "claude-sonnet-5-5"


settings = Settings()
