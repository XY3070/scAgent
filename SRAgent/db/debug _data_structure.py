import os
import sys
from dotenv import load_dotenv
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SRAgent.db.get import get_prefiltered_datasets_functional
from SRAgent.db.connect import db_connect

load_dotenv()
os.environ["DYNACONF"] = "test"

def debug_data_structure():
    try:
        with db_connect() as conn:
            print("=== Debugging Data Structure ===")
            
            # 获取少量数据进行分析
            result_df = get_prefiltered_datasets_functional(
                conn=conn,
                organisms=["human"],
                search_term="cancer",
                limit=3  # 只取3条记录进行调试
            )
            
            if result_df.empty:
                print("❌ No data returned")
                return
            
            print(f"📊 Got {len(result_df)} records")
            print(f"📋 Available columns: {list(result_df.columns)}")
            print("\n" + "="*80)
            
            # 显示前几条记录的详细信息
            for idx, row in result_df.iterrows():
                print(f"\n📄 Record {idx + 1}:")
                print("-" * 40)
                
                # 显示所有字段及其值
                for col in result_df.columns:
                    value = row[col]
                    if pd.isna(value):
                        continue
                    if isinstance(value, str) and value.strip():
                        print(f"  {col}: {value}")
                    elif not isinstance(value, str) and value is not None:
                        print(f"  {col}: {value}")
                
                print("-" * 40)
                
                # 特别关注可能包含项目ID的字段
                project_candidates = []
                for col in result_df.columns:
                    value = str(row[col]) if not pd.isna(row[col]) else ""
                    if any(prefix in value.upper() for prefix in ['GSE', 'PRJNA', 'ENA-STUDY', 'SRP', 'ERP', 'DRP']):
                        project_candidates.append((col, value))
                
                if project_candidates:
                    print(f"🎯 Potential project ID fields for record {idx + 1}:")
                    for col, value in project_candidates:
                        print(f"  ✓ {col}: {value}")
                else:
                    print(f"⚠️  No obvious project ID found in record {idx + 1}")
            
            print("\n" + "="*80)
            print("🔍 Summary of all unique values in key fields:")
            
            # 检查可能包含项目ID的字段
            key_fields = ['sra_ID', 'run_alias', 'experiment_alias', 'sample_alias',
                         'study_alias', 'submission_alias', 'gsm_title']
            
            for field in key_fields:
                if field in result_df.columns:
                    unique_vals = result_df[field].dropna().unique()
                    if len(unique_vals) > 0:
                        print(f"\n🔑 {field}:")
                        for val in unique_vals[:5]:  # 只显示前5个
                            print(f"  - {val}")
                        if len(unique_vals) > 5:
                            print(f"  ... and {len(unique_vals) - 5} more")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_structure()