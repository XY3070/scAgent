from typing import Optional
import os
from typing import Optional
from SRAgent.agents.utils import load_settings

class Config:
    """
    The configuration class for SRAgent.
    """

    def __init__(self):
        # Load the configuration from settings.yml
        config_settings = load_settings()

        # Database settings
        self.DB_HOST: str = os.getenv("SRA_AGENT_DB_HOST", config_settings.get('db_host', config_settings.get('prod', {}).get('db_host', '')))
        self.DB_NAME: str = os.getenv("SRA_AGENT_DB_NAME", config_settings.get('db_name', config_settings.get('prod', {}).get('db_name')))
        self.DB_USER: str = os.getenv("SRA_AGENT_DB_USER", config_settings.get('db_user', config_settings.get('prod', {}).get('db_user', '')))
        self.DB_PASSWORD: str = os.getenv("SRA_AGENT_DB_PASSWORD", config_settings.get('db_password', config_settings.get('prod', {}).get('db_password', '')))
        self.DB_PORT: int = int(os.getenv("SRA_AGENT_DB_PORT", config_settings.get('db_port', config_settings.get('prod', {}).get('db_port', 5432))))
        self.DB_TIMEOUT: int = int(os.getenv("SRA_AGENT_DB_TIMEOUT", config_settings.get('db_timeout', config_settings.get('prod', {}).get('db_timeout', 300))))

        self.QWEN_API_BASE: Optional[str] = os.getenv("SRA_AGENT_QWEN_API_BASE", config_settings.get("qwen_api_base"))

        # Entrez settings (still get from environment variables, as settings.yml does not have these configurations)
        self.ENTREZ_EMAIL: Optional[str] = os.getenv("ENTREZ_EMAIL")
        self.ENTREZ_API_KEY: Optional[str] = os.getenv("ENTREZ_API_KEY")

        # New addition: control the switch for online access
        self.ONLINE_ACCESS_ENABLED = os.getenv("SRA_ONLINE_ACCESS_ENABLED", "False").lower() == "true"

        # New addition: control the switch for using local database
        self.USE_LOCAL_DB = os.getenv("SRA_USE_LOCAL_DB", "False").lower() == "true"

        # Other settings (add as needed)
        self.DYNACONF_ENV: str = os.getenv("DYNACONF_ENV", "prod")

# Instantiate the configuration class, so it can be directly imported and used
settings = Config()

# Entrez ID extraction prompt constants
ENTREZ_ID_EXTRACTION_PROMPT_PREFIX = "You are a helpful assistant for a bioinformatics researcher."
ENTREZ_ID_EXTRACTION_PROMPT_TASKS = """
# Tasks
 - Extract Entrez IDs (e.g., 19007785 or 27176348) from the message below.
    - If you cannot find any Entrez IDs, do not provide any accessions.
    - Entrez IDs may be referred to as 'database IDs' or 'accession numbers'.
 - Extract the database name (e.g., GEO, SRA, etc.)
   - If you cannot find the database name, do not provide any database name.
   - GEO should be formatted as 'gds'
   - SRA should be formatted as 'sra'"""
ENTREZ_ID_EXTRACTION_PROMPT_MESSAGE_START = "#-- START OF MESSAGE --#"
ENTREZ_ID_EXTRACTION_PROMPT_MESSAGE_END = "#-- END OF MESSAGE --#"
ENTREZ_ID_EXTRACTION_PROMPT_RETRY_SUFFIX = "If no valid Entrez IDs or database are found, return empty values."