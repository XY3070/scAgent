from typing import Optional
import os

class Config:
    """Configuration class for SRAgent."""

    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_NAME: str = os.getenv("DB_NAME", "your_database_name")
    DB_USER: str = os.getenv("DB_USER", "your_username")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "your_password")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_TIMEOUT: int = int(os.getenv("DB_TIMEOUT", 10))

    # Entrez settings
    ENTREZ_EMAIL: Optional[str] = os.getenv("ENTREZ_EMAIL")
    ENTREZ_API_KEY: Optional[str] = os.getenv("ENTREZ_API_KEY")

    # Other settings (add as needed)
    DYNACONF_ENV: str = os.getenv("DYNACONF", "prod")

    def __init__(self):
        # Add any validation or initialization logic here
        pass

# Instantiate the config to be imported directly
settings = Config()