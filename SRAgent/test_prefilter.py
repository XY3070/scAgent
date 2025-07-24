import asyncio
import os
import sys
# 添加项目根目录到 PYTHONPATH（根据你的目录结构调整）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/ssd2/xuyuan/SRAgent/SRAgent')))
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入模块
from SRAgent.db.connect import db_connect
from SRAgent.db.get import get_prefiltered_datasets_from_local_db

async def test_prefilter():
    with db_connect() as conn:
        print("✅ 数据库连接成功")

        # 测试参数
        organisms = ["human"]
        min_date = "2020/01/01"
        max_date = "2025/07/23"
        search_term = "cancer"  # 测试关键词
        limit = 100  # 限制返回最多 100 条
        print(f"🔍 正在预筛选数据：物种={organisms}, 时间={min_date}~{max_date}, 关键词='{search_term}'")

        # 调用异步函数
        results = await get_prefiltered_datasets_from_local_db(
            conn=conn,
            organisms=organisms,
            min_date=min_date,
            max_date=max_date,
            search_term=search_term,
            limit=limit
        )

        # 输出结果
        print(f"📊 共找到 {len(results)} 条匹配数据")
        for idx, record in enumerate(results):
            print(f"{idx+1}. {record['srx_id']} - {record['study_title']}")
        return results

if __name__ == "__main__":
    asyncio.run(test_prefilter())
