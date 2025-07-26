import asyncio
import os
import sys
# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/ssd2/xuyuan/SRAgent/SRAgent')))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入模块
from SRAgent.db.connect import db_connect
from SRAgent.db.get import get_prefiltered_datasets_from_local_db

async def test_and_display_results():
    """测试并显示详细结果"""
    
    print("🚀 测试预筛选函数并显示详细结果...")
    
    with db_connect() as conn:
        print("✅ 数据库连接成功")

        # 测试参数
        organisms = ["human"]
        min_date = "2010-01-01"
        max_date = "2025-07-23"
        search_term = "cancer"
        limit = 5  # 减少到5条，方便查看详细信息
        
        print(f"🔍 测试参数:")
        print(f"  - 物种: {organisms}")
        print(f"  - 时间范围: {min_date} ~ {max_date}")
        print(f"  - 搜索关键词: '{search_term}'")
        print(f"  - 限制数量: {limit}")
        print()

        try:
            from SRAgent.db.get import get_prefiltered_datasets_from_local_db
            
            results = await get_prefiltered_datasets_from_local_db(
                conn=conn,
                organisms=organisms,
                min_date=min_date,
                max_date=max_date,
                search_term=search_term,
                limit=limit
            )
            
            print(f"\n📊 成功找到 {len(results)} 条记录")
            print("="*80)
            
            if results:
                for idx, record in enumerate(results):
                    print(f"\n📄 记录 {idx+1}:")
                    print(f"   SRA ID: {record.get('sra_ID', 'N/A')}")
                    print(f"   标题: {record.get('study_title', 'No title')}")
                    print(f"   物种: {record.get('scientific_name', 'N/A')}")
                    print(f"   策略: {record.get('library_strategy', 'N/A')}")
                    print(f"   技术: {record.get('technology', 'N/A')}")
                    print(f"   GSE标题: {record.get('gse_title', 'N/A')}")
                    print(f"   提交日期: {record.get('gsm_submission_date', 'N/A')}")
                    
                    # 显示摘要的前200个字符
                    summary = record.get('summary', '')
                    if summary and summary != 'N/A':
                        print(f"   摘要: {summary[:200]}...")
                    
                    print("-" * 60)
                
                # 显示所有可用字段
                print(f"\n🔍 第一条记录的所有字段:")
                for key, value in results[0].items():
                    if value and str(value).strip() and str(value) != 'None':
                        print(f"   {key}: {str(value)[:100]}...")
                
            return results
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return []

async def test_different_keywords():
    """测试不同的搜索关键词"""
    
    print("\n" + "="*80)
    print("🧪 测试不同的搜索关键词")
    print("="*80)
    
    keywords = ["RNA", "sequencing", "cell", "human", "gene"]
    
    with db_connect() as conn:
        from SRAgent.db.get import get_prefiltered_datasets_from_local_db
        
        for keyword in keywords:
            print(f"\n🔍 搜索关键词: '{keyword}'")
            try:
                results = await get_prefiltered_datasets_from_local_db(
                    conn=conn,
                    organisms=["human"],
                    min_date="2020-01-01",
                    max_date="2025-07-23",
                    search_term=keyword,
                    limit=3
                )
                print(f"   ✅ 找到 {len(results)} 条记录")
                
                if results:
                    for i, record in enumerate(results[:2]):  # 只显示前2条
                        title = record.get('study_title', 'No title')
                        print(f"   {i+1}. {title[:100]}...")
                        
            except Exception as e:
                print(f"   ❌ 搜索 '{keyword}' 失败: {e}")

async def test_no_filters():
    """测试无筛选条件的情况"""
    
    print("\n" + "="*80)
    print("🧪 测试无筛选条件（获取最新数据）")
    print("="*80)
    
    with db_connect() as conn:
        from SRAgent.db.get import get_prefiltered_datasets_from_local_db
        
        try:
            results = await get_prefiltered_datasets_from_local_db(
                conn=conn,
                organisms=[],  # 无物种限制
                min_date="",   # 无日期限制
                max_date="",
                search_term="", # 无关键词
                limit=5
            )
            
            print(f"✅ 无筛选条件找到 {len(results)} 条记录")
            
            if results:
                print("最新的几条记录:")
                for i, record in enumerate(results):
                    print(f"{i+1}. {record.get('study_title', 'No title')[:80]}...")
                    print(f"   日期: {record.get('gsm_submission_date', 'N/A')}")
                    
        except Exception as e:
            print(f"❌ 无筛选测试失败: {e}")

async def main():
    """主测试函数"""
    
    print("="*80)
    print("🎯 全面测试预筛选函数")
    print("="*80)
    
    # 主要功能测试
    results = await test_and_display_results()
    
    # 不同关键词测试
    await test_different_keywords()
    
    # 无筛选测试
    await test_no_filters()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
    