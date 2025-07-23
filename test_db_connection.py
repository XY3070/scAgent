import os
from dotenv import load_dotenv
from SRAgent.db.connect import db_connect

# 加载环境变量
load_dotenv(override=True)

# 设置DYNACONF_ENV为test，以确保使用测试环境的配置
os.environ["DYNACONF_ENV"] = "test"

try:
    with db_connect() as conn:
        print("成功连接到数据库！")
        # 可以在这里添加一个简单的查询来验证表是否存在
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM merged.sra_geo_ft LIMIT 1;")
            print("成功查询 merged.sra_geo_ft 表。")
except Exception as e:
    print(f"连接数据库或查询表时发生错误: {e}")