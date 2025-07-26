import asyncio
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from SRAgent.db.connect import db_connect

def execute_query_with_cursor_debug(conn, query, params):
    """调试版本的查询执行函数"""
    print(f"Query length: {len(query)}")
    print(f"Params length: {len(params)}")
    print(f"Query count of %s: {query.count('%s')}")
    
    # 检查参数类型
    for i, param in enumerate(params[:10]):  # 只检查前10个参数
        print(f"Param {i}: type={type(param)}, value={param}")
    
    try:
        cursor = conn.cursor()
        print("Cursor created successfully")
        
        print("Executing query...")
        cursor.execute(query, params)
        print("Query executed successfully")
        
        if cursor.description is None:
            print("No description - query returned no results")
            cursor.close()
            return []
        
        print("Getting column names...")
        colnames = [desc[0] for desc in cursor.description]
        print(f"Column names: {colnames}")
        
        print("Fetching results...")
        results = cursor.fetchall()
        print(f"Found {len(results)} rows with {len(colnames)} columns")
        
        cursor.close()
        return results
            
    except Exception as e:
        print(f"Query execution failed: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        try:
            cursor.close()
        except:
            pass
        return []

async def simple_test():
    """简化的测试，逐步增加复杂度"""
    with db_connect() as conn:
        print("✅ 数据库连接成功")
        
        # 检查连接状态
        print(f"连接状态: {conn.status}")
        print(f"连接信息: {conn.get_dsn_parameters()}")
        
        # 测试连接是否正常工作
        try:
            with conn.cursor() as test_cursor:
                test_cursor.execute("SELECT 1 as test")
                result = test_cursor.fetchone()
                print(f"连接测试成功: {result}")
        except Exception as e:
            print(f"连接测试失败: {e}")
            return
        
        # 测试1: 最简单的查询
        print("\n=== 测试1: 简单查询 ===")
        simple_query = 'SELECT "sra_ID", study_title FROM merged.sra_geo_ft LIMIT 5'
        results = execute_query_with_cursor_debug(conn, simple_query, ())
        if results:
            print(f"简单查询成功，找到 {len(results)} 条记录")
            print(f"第一条记录: {results[0]}")
        
        # 测试2: 带一个参数的查询
        print("\n=== 测试2: 带参数查询 ===")
        param_query = 'SELECT "sra_ID", study_title FROM merged.sra_geo_ft WHERE study_title ILIKE %s LIMIT 5'
        results = execute_query_with_cursor_debug(conn, param_query, ('%cancer%',))
        if results:
            print(f"参数查询成功，找到 {len(results)} 条记录")
        
        # 测试3: 检查有问题的字段
        print("\n=== 测试3: 检查字段是否存在 ===")
        field_query = '''
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'merged' AND table_name = 'sra_geo_ft'
        ORDER BY column_name
        '''
        results = execute_query_with_cursor_debug(conn, field_query, ())
        if results:
            print("表中的字段:")
            for row in results[:20]:  # 显示前20个字段
                print(f"  - {row[0]}")
        
        # 测试4: 简化的单细胞筛选
        print("\n=== 测试4: 简化单细胞筛选 ===")
        sc_query = '''
        SELECT "sra_ID", study_title, library_strategy
        FROM merged.sra_geo_ft 
        WHERE library_strategy ILIKE %s OR library_strategy ILIKE %s
        LIMIT 5
        '''
        results = execute_query_with_cursor_debug(conn, sc_query, ('%scRNA%', '%single cell%'))
        if results:
            print(f"单细胞筛选成功，找到 {len(results)} 条记录")

if __name__ == "__main__":
    asyncio.run(simple_test())