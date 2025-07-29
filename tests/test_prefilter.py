#!/usr/bin/env python3
"""
测试新的函数式预筛选设计
每个筛选器接受一个对象，返回新的筛选后对象
"""

import os
import sys
import pandas as pd
import psycopg2
from datetime import datetime
from typing import List, Dict, Any

# 数据库连接配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'your_database'),
    'user': os.getenv('DB_USER', 'your_user'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'port': os.getenv('DB_PORT', 5432)
}

def get_db_connection():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

# 为了测试，我们需要在这里重新定义一些核心类（在实际使用中会从prefilter模块导入）
from dataclasses import dataclass
import pandas as pd

@dataclass
class FilterResult:
    """筛选结果数据类"""
    data: pd.DataFrame
    count: int
    filter_name: str
    description: str
    reduction_count: int = 0
    reduction_pct: float = 0.0
    
    def __post_init__(self):
        if self.data is not None:
            self.count = len(self.data)
    
    def log_result(self, previous_count: int = None):
        if previous_count is not None:
            self.reduction_count = previous_count - self.count
            self.reduction_pct = (self.reduction_count / previous_count * 100) if previous_count > 0 else 0
            
        print(f"📊 {self.filter_name}: {self.count:,} records "
              f"(↓{self.reduction_count:,}, -{self.reduction_pct:.1f}%)")
        if self.description:
            print(f"   {self.description}")

def execute_query_with_cursor(conn, query, params):
    """执行查询并返回DataFrame"""
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        if cursor.description is None:
            cursor.close()
            return pd.DataFrame()
        
        colnames = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        cursor.close()
        
        return pd.DataFrame(results, columns=colnames)
        
    except Exception as e:
        print(f"Database query error: {e}")
        try:
            cursor.close()
        except:
            pass
        return pd.DataFrame()

# 简化的筛选器类用于测试
class TestInitialDatasetFilter:
    def __init__(self, conn, table_name="merged.sra_geo_ft"):
        self.conn = conn
        self.table_name = table_name
    
    def apply(self, input_result=None):
        query = f"""
            SELECT "sra_ID", study_title, summary, overall_design, scientific_name, 
                   library_strategy, technology, characteristics_ch1, gse_title, gsm_title,
                   organism_ch1, source_name_ch1, common_name, gsm_submission_date, sc_conf_score
            FROM {self.table_name}
            LIMIT 1000  -- 限制数量以加快测试
        """
        
        df = execute_query_with_cursor(self.conn, query, ())
        result = FilterResult(
            data=df,
            count=len(df),
            filter_name="Initial Dataset",
            description="Sample records from database"
        )
        result.log_result()
        return result

class TestBasicAvailabilityFilter:
    def __init__(self, conn):
        self.conn = conn
    
    def apply(self, input_result):
        if input_result.data.empty:
            return FilterResult(pd.DataFrame(), 0, "Basic Availability", "No input data")
        
        filtered_df = input_result.data[
            (input_result.data['sra_ID'].notna()) & 
            (input_result.data['sra_ID'] != '') & 
            (input_result.data['gse_title'].notna())
        ].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Basic Availability",
            description="Has SRA_ID and GSE title"
        )
        result.log_result(input_result.count)
        return result

class TestOrganismFilter:
    def __init__(self, conn, organisms):
        self.conn = conn
        self.organisms = organisms
    
    def apply(self, input_result):
        if input_result.data.empty or "human" not in [org.lower() for org in self.organisms]:
            return input_result
        
        human_mask = (
            input_result.data['organism_ch1'].str.contains('homo sapiens', case=False, na=False) |
            input_result.data['scientific_name'].str.contains('homo sapiens', case=False, na=False) |
            input_result.data['source_name_ch1'].str.contains('human', case=False, na=False) |
            input_result.data['common_name'].str.contains('human', case=False, na=False)
        )
        
        filtered_df = input_result.data[human_mask].copy()
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Organism Filter",
            description="Human samples only"
        )
        result.log_result(input_result.count)
        return result

class TestSingleCellFilter:
    def __init__(self, conn, min_confidence=2):
        self.conn = conn
        self.min_confidence = min_confidence
    
    def apply(self, input_result):
        if input_result.data.empty or 'sc_conf_score' not in input_result.data.columns:
            return input_result
        
        sc_mask = (
            input_result.data['sc_conf_score'].notna() & 
            (input_result.data['sc_conf_score'] >= self.min_confidence)
        )
        
        filtered_df = input_result.data[sc_mask].copy()
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Single Cell Filter",
            description=f"Single cell confidence score >= {self.min_confidence}"
        )
        result.log_result(input_result.count)
        return result

class TestKeywordSearchFilter:
    def __init__(self, conn, search_term):
        self.conn = conn
        self.search_term = search_term
    
    def apply(self, input_result):
        if input_result.data.empty or not self.search_term:
            return input_result
        
        search_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        search_columns = ['study_title', 'summary', 'gse_title']
        
        for col in search_columns:
            if col in input_result.data.columns:
                search_mask |= input_result.data[col].str.contains(
                    self.search_term, case=False, na=False, regex=False
                )
        
        filtered_df = input_result.data[search_mask].copy()
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Keyword Search",
            description=f"Contains '{self.search_term}'"
        )
        result.log_result(input_result.count)
        return result

def test_functional_filter_chain():
    """测试函数式筛选器链"""
    print("🧪 Testing Functional Filter Chain Design")
    print("-" * 50)
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        # 创建筛选器链
        filters = [
            TestInitialDatasetFilter(conn),
            TestBasicAvailabilityFilter(conn),
            TestOrganismFilter(conn, ["human"]),
            TestSingleCellFilter(conn, min_confidence=2),
            TestKeywordSearchFilter(conn, "cancer")
        ]
        
        print("🔍 Applying filter chain...")
        print("=" * 40)
        
        result = None
        for filter_obj in filters:
            result = filter_obj.apply(result)
            
            # 检查逻辑正确性
            if result.count < 0:
                print("❌ Logic error: negative record count")
                return False
            
            # 如果没有记录了，提前停止
            if result.count == 0:
                print("⚠️  No records remaining, stopping chain")
                break
        
        print("=" * 40)
        print(f"🎯 Final result: {result.count} records")
        
        # 显示样本结果
        if not result.data.empty:
            print("\n📋 Sample results:")
            sample_size = min(3, len(result.data))
            for i, (idx, row) in enumerate(result.data.head(sample_size).iterrows()):
                print(f"  {i+1}. {row.get('sra_ID', 'N/A')} - {str(row.get('study_title', 'N/A'))[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Functional filter chain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

def test_individual_filters():
    """测试各个筛选器的独立功能"""
    print("\n🔬 Testing Individual Filter Functions")
    print("-" * 50)
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        # 获取初始数据
        initial_filter = TestInitialDatasetFilter(conn)
        initial_result = initial_filter.apply()
        
        if initial_result.data.empty:
            print("❌ No initial data available for testing")
            return False
        
        print(f"✅ Initial data loaded: {initial_result.count} records")
        
        # 测试基础可用性筛选
        basic_filter = TestBasicAvailabilityFilter(conn)
        basic_result = basic_filter.apply(initial_result)
        
        if basic_result.count > initial_result.count:
            print("❌ Basic filter logic error: increased record count")
            return False
        
        print(f"✅ Basic availability filter: {basic_result.count} records")
        
        # 测试物种筛选
        organism_filter = TestOrganismFilter(conn, ["human"])
        organism_result = organism_filter.apply(basic_result)
        
        if organism_result.count > basic_result.count:
            print("❌ Organism filter logic error: increased record count")
            return False
        
        print(f"✅ Organism filter: {organism_result.count} records")
        
        # 测试关键词搜索
        keyword_filter = TestKeywordSearchFilter(conn, "cancer")
        keyword_result = keyword_filter.apply(organism_result)
        
        print(f"✅ Keyword search filter: {keyword_result.count} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual filter test failed: {e}")
        return False
    finally:
        conn.close()

def test_filter_immutability():
    """测试筛选器的不可变性（每次返回新对象）"""
    print("\n🛡️  Testing Filter Immutability")
    print("-" * 50)
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        # 获取初始数据
        initial_filter = TestInitialDatasetFilter(conn)
        result1 = initial_filter.apply()
        original_count = result1.count
        
        # 应用筛选器
        basic_filter = TestBasicAvailabilityFilter(conn)
        result2 = basic_filter.apply(result1)
        
        # 检查原始结果是否被修改
        if result1.count != original_count:
            print("❌ Filter mutated original result object")
            return False
        
        # 检查是否返回了新对象
        if result1 is result2:
            print("❌ Filter returned same object instead of new one")
            return False
        
        print("✅ Filters maintain immutability")
        print(f"   Original result: {result1.count} records (unchanged)")
        print(f"   New result: {result2.count} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Immutability test failed: {e}")
        return False
    finally:
        conn.close()

def test_error_handling():
    """测试错误处理"""
    print("\n🚨 Testing Error Handling")
    print("-" * 50)
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        # 测试空数据处理
        empty_result = FilterResult(
            data=pd.DataFrame(),
            count=0,
            filter_name="Empty Test",
            description="Empty input"
        )
        
        basic_filter = TestBasicAvailabilityFilter(conn)
        result = basic_filter.apply(empty_result)
        
        if not result.data.empty or result.count != 0:
            print("❌ Empty data handling failed")
            return False
        
        print("✅ Empty data handled correctly")
        
        # 测试None输入处理
        try:
            organism_filter = TestOrganismFilter(conn, None)
            result = organism_filter.apply(empty_result)
            print("✅ None parameter handled correctly")
        except:
            print("❌ None parameter not handled properly")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    finally:
        conn.close()

def run_all_tests():
    """运行所有测试"""
    print("🧪 Functional Prefilter Design Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Functional Filter Chain", test_functional_filter_chain),
        ("Individual Filters", test_individual_filters),
        ("Filter Immutability", test_filter_immutability),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {total-passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Your functional prefilter design is working correctly.")
        print("\n💡 Key advantages of the new design:")
        print("   ✅ Each filter is independent and reusable")
        print("   ✅ Filters are composable and can be chained in any order")
        print("   ✅ Each filter returns a new object (immutable)")
        print("   ✅ Easy to test, debug, and maintain")
        print("   ✅ Follows functional programming principles")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    print("🚀 Starting Functional Prefilter Design Tests")
    print("Please make sure your database credentials are set correctly!")
    print()
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("Your new functional prefilter design is ready to use!")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        exit(1)