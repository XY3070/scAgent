import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
from psycopg2.extensions import connection
import logging
from datetime import datetime  

# import prefilter module
# Fix relative import issue
try:
    # Try relative import first (when used as module)
    from .prefilter import (
        FilterResult, 
        create_filter_chain, 
        apply_filter_chain,
        InitialDatasetFilter,
        BasicAvailabilityFilter,
        OrganismFilter,
        SingleCellFilter,
        SequencingStrategyFilter,
        CancerStatusFilter,
        TissueSourceFilter,
        KeywordSearchFilter,
        LimitFilter
    )
    from .utils import execute_query
except ImportError:
    # Fallback for direct execution
    try:
        from prefilter import (
            FilterResult, 
            create_filter_chain, 
            apply_filter_chain,
            InitialDatasetFilter,
            BasicAvailabilityFilter,
            OrganismFilter,
            SingleCellFilter,
            SequencingStrategyFilter,
            CancerStatusFilter,
            TissueSourceFilter,
            KeywordSearchFilter,
            LimitFilter
        )
        try:
            from utils import execute_query
        except ImportError:
            from .utils import execute_query
    except ImportError:
        print("❌ Cannot import prefilter module. Please check your setup.")
        sys.exit(1)

logger = logging.getLogger(__name__)


# Prefilter function, using independent filter module
def get_prefiltered_datasets_functional(
    conn: connection,
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 100,
    min_sc_confidence: int = 2,
    create_temp_table: bool = False,
    temp_table_name: str = "temp_prefiltered_results"
) -> pd.DataFrame:
    """
    Prefilter datasets using a functional prefiltering approach, where each filter
    accepts an object and returns a new filtered object.
    
    Args:
        conn: Database connection
        organisms: List of organisms, default ["human"]
        search_term: Search keyword
        limit: Limit on the number of records returned
        min_sc_confidence: Minimum single-cell confidence score
        create_temp_table: Whether to create a temporary table
        temp_table_name: Name of the temporary table
    
    Returns:
        Prefiltered DataFrame
    """
    try:
        # Create filter chain
        filter_chain = create_filter_chain(
            conn=conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit,
            min_sc_confidence=min_sc_confidence
        )
        
        # Apply filter chain
        final_result = apply_filter_chain(filter_chain)
        
        # If needed, create temporary table
        if create_temp_table and not final_result.data.empty:
            create_temporary_table(conn, final_result.data, temp_table_name)
        
        return final_result.data
        
    except Exception as e:
        logger.error(f"Prefiltering failed: {e}")
        return pd.DataFrame()

def get_prefiltered_datasets_custom_chain(
    conn: connection,
    custom_filters: List[str],
    filter_params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Prefilter datasets using a custom filter chain.
    
    Args:
        conn: Database connection
        custom_filters: List of custom filter names
        filter_params: Dictionary of filter parameters
    
    Returns:
        Prefiltered DataFrame
    """
    if filter_params is None:
        filter_params = {}
    
    # Map filter names to classes
    filter_map = {
        'initial': InitialDatasetFilter,
        'basic': BasicAvailabilityFilter,
        'organism': OrganismFilter,
        'single_cell': SingleCellFilter,
        'sequencing': SequencingStrategyFilter,
        'cancer': CancerStatusFilter,
        'tissue': TissueSourceFilter,
        'keyword': KeywordSearchFilter,
        'limit': LimitFilter
    }
    
    try:
        # Build custom filter chain
        filter_chain = []
        result = None
        
        for filter_name in custom_filters:
            if filter_name not in filter_map:
                logger.warning(f"Unknown filter: {filter_name}")
                continue
            
            filter_class = filter_map[filter_name]
            
            # Create instance based on filter type
            if filter_name == 'organism':
                filter_obj = filter_class(conn, filter_params.get('organisms', ['human']))
            elif filter_name == 'single_cell':
                filter_obj = filter_class(conn, filter_params.get('min_sc_confidence', 2))
            elif filter_name == 'keyword':
                filter_obj = filter_class(conn, filter_params.get('search_term'))
            elif filter_name == 'limit':
                filter_obj = filter_class(conn, filter_params.get('limit', 100))
            else:
                filter_obj = filter_class(conn)
            
            # Apply filter
            result = filter_obj.apply(result)
            
            if result.count == 0:
                logger.warning("No records remaining after filter: " + filter_name)
                break
        
        return result.data if result else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Custom chain filtering failed: {e}")
        return pd.DataFrame()

def create_temporary_table(conn: connection, df: pd.DataFrame, table_name: str):
    """
    Create a temporary table and insert prefiltered results.
    
    Args:
        conn: Database connection
        df: DataFrame to insert
        table_name: Name of the temporary table
    """
    try:
        cursor = conn.cursor()
        
        # Delete existing temporary table
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create temporary table structure (based on DataFrame columns)
        columns_def = []
        for col in df.columns:
            # Simple type mapping, can be expanded as needed
            if df[col].dtype == 'object':
                columns_def.append(f'"{col}" TEXT')
            elif df[col].dtype == 'int64':
                columns_def.append(f'"{col}" INTEGER')
            elif df[col].dtype == 'float64':
                columns_def.append(f'"{col}" REAL')
            else:
                columns_def.append(f'"{col}" TEXT')
        
        create_sql = f"CREATE TEMP TABLE {table_name} ({', '.join(columns_def)})"
        cursor.execute(create_sql)
        
        # Insert data
        if not df.empty:
            # Prepare insert statement
            placeholders = ', '.join(['%s'] * len(df.columns))
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
            
            # Convert DataFrame to list of tuples
            data_tuples = [tuple(row) for row in df.values]
            
            # Batch insert
            cursor.executemany(insert_sql, data_tuples)
        
        conn.commit()
        cursor.close()
        
        logger.info(f"Created temporary table '{table_name}' with {len(df)} records")
        
    except Exception as e:
        logger.error(f"Failed to create temporary table: {e}")
        try:
            cursor.close()
        except:
            pass

# Retain original function's compatibility version
async def get_prefiltered_datasets_from_local_db(
    conn,
    organisms: list,
    search_term: str,
    limit: int = 100
) -> list:
    """
    Compatibility version of original prefiltering function, now using new functional filters
    Maintains compatibility with existing code
    """
    try:
        # Call new functional prefiltering method   
        result_df = get_prefiltered_datasets_functional(
            conn=conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit
        )
        
        if result_df.empty:
            logger.info("No records found with current filters")
            return []
        
        # Convert to original format (list of dictionaries)
        records = result_df.to_dict(orient='records')
        logger.info(f"Successfully found {len(records)} records")
        return records
        
    except Exception as e:
        logger.error(f"Prefiltering error: {e}")
        return []

def check_table_structure(conn):
    """
    Check the table structure and confirm which fields actually exist
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'merged' AND table_name = 'sra_geo_ft'
            ORDER BY column_name
        """)
        
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        print("📊 Actual fields existing in the database table:")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        
        return columns
        
    except Exception as e:
        print(f"❌ Failed to check table structure: {e}")
        return []

# Example usage function
def example_usage():
    """
    Show how to use the new prefiltering and export system
    """
    from dotenv import load_dotenv
    try:
        from connect import db_connect
    except ImportError:
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            sys.path.insert(0, project_root)
            from SRAgent.db.connect import db_connect
        except ImportError:
            print("Cannot import db_connect. Please check import paths.")
            return
    
    load_dotenv()
    
    with db_connect() as conn:
        print("=== Example 1: JSON Export with Full Categorization ===")
        result1 = export_prefiltered_datasets_to_json(
            conn=conn,
            output_path="exports/example_datasets.json",
            organisms=["human"],
            search_term="cancer",
            limit=50,
            include_categorization=True,
            include_grouping=True
        )
        print(f"JSON Export Result: {result1['status']}")
        
        print("\n=== Example 2: SQLite Export ===")
        result2 = export_prefiltered_datasets_to_sqlite(
            conn=conn,
            output_path="exports/example_datasets.db",
            organisms=["human"],
            search_term="brain",
            limit=30,
            create_categorized_tables=True
        )
        print(f"SQLite Export Result: {result2['status']}")
        
        print("\n=== Example 3: Classify-Ready Export ===")
        result3 = create_classify_ready_export(
            conn=conn,
            output_dir="exports/classify_ready",
            organisms=["human"],
            search_term="single cell",
            limit=100,
            export_format="json"
        )
        print(f"Classify-Ready Export Result: {result3['status']}")

        # export function
        print("\n=== Export Examples ===")
        from .export import create_classify_ready_export
    
        result = create_classify_ready_export(
            conn=conn,
            output_dir="exports/classify_ready",
            organisms=["human"],
            export_format="json"
        )
        print(f"Export Result: {result['status']}")


# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        from SRAgent.db.connect import db_connect
    except ImportError:
        print("Cannot import db_connect. Please check import paths.")
        sys.exit(1)
    
    def test_export_functionality(conn, df):
        """临时测试导出功能"""
        try:
            import json
            from pathlib import Path
            
            if df.empty:
                print("No data to export")
                return {"status": "no_data"}
            
            # 简单的项目分类测试
            def categorize_test(df):
                categorized = {'GSE': [], 'PRJNA': [], 'ena-STUDY': [], 'discarded': []}
                records = df.to_dict(orient='records')
                
                for record in records:
                    project_id = None
                    for field in ['sra_study_accession', 'bioproject_accession', 'study_accession']:
                        if field in record and record[field]:
                            project_id = str(record[field])
                            break
                    
                    if not project_id:
                        categorized['discarded'].append(record)
                        continue
                        
                    if project_id.startswith('GSE'):
                        categorized['GSE'].append(record)
                    elif project_id.startswith('PRJNA'):
                        categorized['PRJNA'].append(record)
                    elif project_id.startswith('ena-STUDY'):
                        categorized['ena-STUDY'].append(record)
                    else:
                        categorized['discarded'].append(record)
                
                return categorized
            
            # 执行分类
            categorized = categorize_test(df)
            
            # 创建导出数据
            export_data = {
                "export_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_records": len(df),
                    "test_export": True,
                    "categorization_stats": {
                        category: len(records) for category, records in categorized.items()
                    }
                },
                "raw_data": df.to_dict(orient='records'),
                "categorized_data": categorized
            }
            
            # 创建输出目录
            output_dir = Path("test_export")
            output_dir.mkdir(exist_ok=True)
            
            # 写入主文件
            with open(output_dir / "test_datasets.json", "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # 为每个类别创建单独文件
            for category, records in categorized.items():
                if records:
                    category_file = output_dir / f"{category.lower()}_projects.json"
                    with open(category_file, "w") as f:
                        json.dump({
                            "category": category,
                            "count": len(records),
                            "projects": records
                        }, f, indent=2, default=str)
            
            print(f"✅ Test export successful!")
            print(f"   📁 Output directory: {output_dir}")
            print(f"   📊 Total records: {len(df)}")
            print(f"   📋 Categorization:")
            for category, records in categorized.items():
                if records:
                    print(f"      - {category}: {len(records)} records")
            
            return {"status": "success", "output_dir": str(output_dir)}
            
        except Exception as e:
            print(f"❌ Test export failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    os.environ["DYNACONF"] = "test"
    try: 
        with db_connect() as conn:
            # Testing prefilter functions
            print("=== Testing prefilter functions ===")
            try:
                result_df = get_prefiltered_datasets_functional(
                    conn=conn,
                    organisms=["human"],
                    search_term="cancer",
                    limit=10
                )
                print(f"Prefilter result: {len(result_df)} records")
            except Exception as e:
                print(f"Prefiltering error: {e}")

            # Testinbg export functions 
            print("\n=== Testing export functions ===")
            try:
            # Plan A: Dynamic import
                import importlib.util
                import sys
                from pathlib import Path
                
                # Get export module path
                current_dir = Path(__file__).parent
                export_dir = current_dir / "export"
                
                if export_dir.exists():
                    sys.path.insert(0, str(current_dir))
                    from export.categorize import create_classify_ready_export
                else:
                    # If not split yet, define a simple test function in the current file
                    print("Export module not yet created, skipping export test")
                    raise ImportError("Export module not found")
                
                export_result = create_classify_ready_export(
                    conn=conn,
                    output_dir="test_export",
                    organisms=["human"],
                    limit=5,
                    export_format="json"
                )
                print(f"Export result: {export_result['status']}")
                
            except ImportError as e:
                print(f"Export module not available yet: {e}")
            except Exception as e:
                print(f"Export error: {e}")

    except Exception as e:
        print(f"Database connection error: {e}")
    