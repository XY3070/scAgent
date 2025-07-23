from typing import Optional
import os
from typing import Optional
from SRAgent.agents.utils import load_settings

class Config:
    """SRAgent的配置类。"""

    def __init__(self):
        # 加载settings.yml中的配置
        config_settings = load_settings()

        # 数据库设置
        self.DB_HOST: str = config_settings.get('db_host', config_settings.get('prod', {}).get('db_host', ''))
        self.DB_NAME: str = config_settings.get('db_name', config_settings.get('prod', {}).get('db_name'))
        self.DB_USER: str = config_settings.get('db_user', config_settings.get('prod', {}).get('db_user', ''))
        self.DB_PASSWORD: str = config_settings.get('db_password', config_settings.get('prod', {}).get('db_password', ''))
        self.DB_PORT: int = config_settings.get('db_port', config_settings.get('prod', {}).get('db_port', 5432))
        self.DB_TIMEOUT: int = config_settings.get('db_timeout', config_settings.get('prod', {}).get('db_timeout', 300))

        # Entrez 设置 (仍然从环境变量获取，因为settings.yml中没有这些配置)
        self.ENTREZ_EMAIL: Optional[str] = os.getenv("ENTREZ_EMAIL")
        self.ENTREZ_API_KEY: Optional[str] = os.getenv("ENTREZ_API_KEY")

        # 新增：控制在线访问的开关
        self.ONLINE_ACCESS_ENABLED = os.getenv("SRA_ONLINE_ACCESS_ENABLED", "False").lower() == "true"

        # 其他设置 (根据需要添加)
        self.DYNACONF_ENV: str = os.getenv("DYNACONF_ENV", "prod")

# 实例化配置，以便直接导入使用
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