from typing import Optional
import os

class Config:
    """SRAgent的配置类。"""

    # 数据库设置
    DB_HOST: str = os.getenv("SRA_AGENT_DB_HOST", "localhost")
    DB_NAME: str = os.getenv("SRA_AGENT_DB_NAME", "your_database_name")
    DB_USER: str = os.getenv("SRA_AGENT_DB_USER", "your_username")
    DB_PASSWORD: str = os.getenv("SRA_AGENT_DB_PASSWORD", "your_password")
    DB_PORT: int = int(os.getenv("SRA_AGENT_DB_PORT", 5432))
    DB_TIMEOUT: int = int(os.getenv("DB_TIMEOUT", 10))

    # Entrez 设置
    ENTREZ_EMAIL: Optional[str] = os.getenv("ENTREZ_EMAIL")
    ENTREZ_API_KEY: Optional[str] = os.getenv("ENTREZ_API_KEY")

    # 其他设置 (根据需要添加)
    DYNACONF_ENV: str = os.getenv("DYNACONF", "prod")

    def __init__(self):
        # 在此处添加任何验证或初始化逻辑
        pass

# 实例化配置，以便直接导入使用
settings = Config()