from typing import Optional
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv() # 加载.env文件中的环境变量

class Config:
    """SRAgent的配置类。"""

    def __init__(self):
        # 数据库设置
        self.DB_HOST: str = os.getenv('DB_HOST', '')
        self.DB_NAME: str = os.getenv('DB_NAME', '')
        self.DB_USER: str = os.getenv('DB_USER', '')
        self.DB_PASSWORD: str = os.getenv('DB_PASSWORD', '')
        self.DB_PORT: int = int(os.getenv('DB_PORT', '5432'))

        # 模型API设置
        self.MODEL_API_URL: str = os.getenv('MODEL_API_URL', '')
        self.MODEL_NAME: str = os.getenv('MODEL_NAME', '')
        self.DB_TIMEOUT: int = int(os.getenv('DB_TIMEOUT', '300'))

        # 新增：控制在线访问的开关
        self.ONLINE_ACCESS_ENABLED = False # 用户明确表示不需要在线访问

        # 其他设置 (根据需要添加)
        self.DYNACONF_ENV: str = os.getenv("DYNACONF_ENV", "prod")

# 实例化配置，以便直接导入使用
settings = Config()