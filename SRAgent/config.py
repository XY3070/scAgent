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
        self.DB_HOST: str = config_settings['DB_HOST']
        self.DB_NAME: str = config_settings['DB_NAME']
        self.DB_USER: str = config_settings['DB_USER']
        self.DB_PASSWORD: str = config_settings['DB_PASSWORD']
        self.DB_PORT: int = config_settings['DB_PORT']
        self.DB_TIMEOUT: int = config_settings['DB_TIMEOUT']

        # Entrez 设置 (仍然从环境变量获取，因为settings.yml中没有这些配置)
        self.ENTREZ_EMAIL: Optional[str] = os.getenv("ENTREZ_EMAIL")
        self.ENTREZ_API_KEY: Optional[str] = os.getenv("ENTREZ_API_KEY")

        # 其他设置 (根据需要添加)
        self.DYNACONF_ENV: str = os.getenv("DYNACONF_ENV", "prod")

# 实例化配置，以便直接导入使用
settings = Config()