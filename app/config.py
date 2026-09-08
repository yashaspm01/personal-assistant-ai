from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "claude"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    embedding_provider: str = "local"

    app_api_key: str = "change_this"

    tmdb_api_key: str = ""
    news_api_key: str = ""
    github_token: str = ""

    # Comma-separated list — add your real frontend URL once deployed.
    # Never use "*" once this API touches real personal data.
    allowed_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:8080"

    chroma_persist_dir: str = "./app/data/chroma"
    sqlite_path: str = "./app/data/app.db"

    class Config:
        env_file = ".env"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
