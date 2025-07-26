import asyncio
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from SRAgent.db.connect import db_connect

async def test_fixed_functions():
    """测试修复后的函数"""
    
    print("="*80)
    print("🔧 测试最终修复版本")
    print("="*80)
    
    with db_connect() as conn:
        print("✅ 数据库连接成功")
        
        # 首先检查表结构
        print("\n📋 步骤1: 检查表结构")
        from SRAgent.db.get import check_table_structure
        available_columns = check_table_structure(conn)
        
        # 然后做最简化测试
        print("\n🧪 步骤2: 最简化测试")
        from SRAgent.db.get import simple_test_query
        simple_results = await simple_test_query(conn)
        
        if not simple_results:
            print("❌ 连最简单的查询都失败了，需要检查数据库连接和数据")
            return
        
        # 测试修复后的主函数
        print("\n📊 步骤3: 测试修复后的主预筛选函数")
        try:
            from SRAgent.db.get import get_prefiltered_datasets_from_local_db
            
            # 测试不同的参数组合
            test_cases = [
                {
                    "name": "基本测试 - 人类+癌症",
                    "params": {
                        "organisms": ["human"],
                        "min_date": "",
                        "max_date": "",
                        "search_term": "cancer",
                        "limit": 5
                    }
                },
                {
                    "name": "只搜索人类数据",
                    "params": {
                        "organisms": ["human"],
                        "min_date": "",
                        "max_date": "",
                        "search_term": "",
                        "limit": 5
                    }
                },
                {
                    "name": "只搜索关键词",
                    "params": {
                        "organisms": [],
                        "min_date": "",
                        "max_date": "",
                        "search_term": "RNA",
                        "limit": 5
                    }
                },
                {
                    "name": "带日期限制",
                    "params": {
                        "organisms": ["human"],
                        "min_date": "2020-01-01",
                        "max_date": "2025-07-23",
                        "search_term": "",
                        "limit": 5
                    }
                }
            ]
            
            for test_case in test_cases:
                print(f"\n  🔍 {test_case['name']}:")
                try:
                    results = await get_prefiltered_datasets_from_local_db(
                        conn=conn,
                        **test_case['params']
                    )
                    
                    print(f"     ✅ 找到 {len(results)} 条记录")
                    
                    if results:
                        # 显示第一条记录的详细信息
                        first_record = results[0]
                        print(f"     📄 第一条记录:")
                        print(f"        SRA ID: {first_record.get('sra_ID', 'N/A')}")
                        print(f"        标题: {first_record.get('study_title', 'N/A')[:80]}...")
                        print(f"        物种: {first_record.get('scientific_name', 'N/A')}")
                        print(f"        策略: {first_record.get('library_strategy', 'N/A')}")
                        
                except Exception as e:
                    print(f"     ❌ 失败: {e}")
                    # 不要因为一个测试失败就停止，继续下一个测试
                    continue
        
        except ImportError as e:
            print(f"❌ 导入函数失败: {e}")
            print("请确保您已经更新了 db/get.py 文件")
        
        # 测试单细胞专用函数
        print("\n🧬 步骤4: 测试单细胞专用函数")
        try:
            from SRAgent.db.get import get_single_cell_datasets_from_local_db
            
            sc_results = await get_single_cell_datasets_from_local_db(
                conn=conn,
                organisms=["human"],
                min_date="",
                max_date="",
                search_term="",
                limit=5
            )
            
            print(f"✅ 单细胞筛选找到 {len(sc_results)} 条记录")
            
            if sc_results:
                for i, record in enumerate(sc_results[:2]):
                    print(f"  {i+1}. {record.get('study_title', 'No title')[:60]}...")
                    print(f"     策略: {record.get('library_strategy', 'N/A')}")
                    print(f"     技术: {record.get('technology', 'N/A')}")
            
        except Exception as e:
            print(f"❌ 单细胞测试失败: {e}")

async def final_validation():
    """最终验证"""
    
    print("\n" + "="*80)
    print("✅ 最终验证")
    print("="*80)
    
    with db_connect() as conn:
        from SRAgent.db.get import get_prefiltered_datasets_from_local_db
        
        print("🎯 执行最终的综合测试...")
        
        try:
            # 使用您最可能在实际应用中使用的参数
            final_results = await get_prefiltered_datasets_from_local_db(
                conn=conn,
                organisms=["human"],
                min_date="2020-01-01",
                max_date="2025-07-23",
                search_term="cancer",
                limit=10
            )
            
            if final_results:
                print(f"🎉 最终测试成功！找到 {len(final_results)} 条记录")
                print("\n📋 结果摘要:")
                for i, record in enumerate(final_results[:3]):
                    print(f"  {i+1}. {record.get('sra_ID', 'N/A')}")
                    print(f"     {record.get('study_title', 'No title')[:70]}...")
                    print()
                    
                print("✅ 您的预筛选函数现在可以正常工作了！")
                
            else:
                print("⚠️ 函数正常执行但没有找到匹配的记录")
                print("这可能是因为数据库中缺少符合所有条件的数据")
                print("建议:")
                print("  1. 尝试放宽搜索条件")
                print("  2. 检查数据库中是否有足够的数据")
                print("  3. 调整关键词搜索")
                
        except Exception as e:
            print(f"❌ 最终测试失败: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """主函数"""
    
    # 测试修复后的函数
    await test_fixed_functions()
    
    # 最终验证
    await final_validation()
    
    print("\n" + "="*80)
    print("🏁 测试完成！")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())