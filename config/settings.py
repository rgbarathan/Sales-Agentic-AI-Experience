"""Configuration settings for the Agentic AI Sales System."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Environment
    environment: str = "development"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    
    # Database
    sqlite_db_path: str = "./data/agentic_sales.db"
    
    # LLM Configuration
    google_api_key: Optional[str] = None
    llm_model: str = "gemini-2.0-flash-exp"
    llm_temperature: float = 0.7
    
    # Agent Configuration
    agent_timeout_seconds: int = 30
    max_agent_retries: int = 3
    
    # MCP Configuration
    mcp_server_port_start: int = 8100
    
    # Mock Data Configuration
    mock_data_seed: int = 42
    enable_mock_delays: bool = True
    mock_delay_ms: int = 500
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Analytics
    enable_analytics: bool = True
    analytics_batch_size: int = 100


# Global settings instance
settings = Settings()
