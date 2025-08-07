import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
from psycopg2.extensions import connection
import logging
from datetime import datetime


# Fix import path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir) # Make project_root a global variable

def setup_imports():
    """Setup proper import paths for the module"""
    # Add paths to sys.path if not already present
    for path in [project_root, parent_dir, current_dir]: # Prioritize project_root
        if path not in sys.path:
            sys.path.append(path)

# Setup imports first
setup_imports()

# Now import modules with proper error handling
def import_required_modules():
    """Import required modules with fallback strategies"""
    modules = {}
    
    # Try importing prefilter_functions module
    try:
        from prefilter_functions import get_prefiltered_datasets_functional
        modules['prefilter_functions'] = {'get_prefiltered_datasets_functional': get_prefiltered_datasets_functional}
    except ImportError as e:
        print(f"❌ Cannot import prefilter_functions module: {e}")
        print("Please ensure prefilter_functions.py is in the same directory or in your Python path")
        return None

    # Try importing prefilter module
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
        modules['prefilter'] = {
            'FilterResult': FilterResult,
            'create_filter_chain': create_filter_chain,
            'apply_filter_chain': apply_filter_chain,
            'filters': {
                'InitialDatasetFilter': InitialDatasetFilter,
                'BasicAvailabilityFilter': BasicAvailabilityFilter,
                'OrganismFilter': OrganismFilter,
                'SingleCellFilter': SingleCellFilter,
                'SequencingStrategyFilter': SequencingStrategyFilter,
                'CancerStatusFilter': CancerStatusFilter,
                'TissueSourceFilter': TissueSourceFilter,
                'KeywordSearchFilter': KeywordSearchFilter,
                'LimitFilter': LimitFilter
            }
        }
    except ImportError as e:
        print(f"❌ Cannot import prefilter module: {e}")
        print("Please ensure prefilter.py is in the same directory or in your Python path")
        return None
    
    # Try importing utils
    try:
        from utils import execute_query
        modules['utils'] = {'execute_query': execute_query}
    except ImportError:
        try:
            # Try alternative import path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            utils_path = os.path.join(current_dir, 'utils.py')
            if os.path.exists(utils_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("utils", utils_path)
                utils_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(utils_module)
                modules['utils'] = {'execute_query': utils_module.execute_query}
            else:
                print("⚠️ Warning: utils module not found, some functionality may be limited")
                modules['utils'] = None
        except Exception as e:
            print(f"⚠️ Warning: Could not import utils: {e}")
            modules['utils'] = None
    
    # Try importing export modules
    try:
        # Try to import enhanced_metadata
        try:
            from enhanced_metadata import EnhancedMetadataExtractor, enhance_existing_categorize_workflow
            modules['enhanced_metadata'] = {
                'EnhancedMetadataExtractor': EnhancedMetadataExtractor,
                'enhance_existing_categorize_workflow': enhance_existing_categorize_workflow
            }
        except ImportError:
            print("⚠️ Warning: enhanced_metadata module not found")
            modules['enhanced_metadata'] = {}
        
        # Try to import categorization_logic module
        try:
            from categorization_logic import categorize_datasets_by_project

            modules['categorize'] = {'categorize_datasets_by_project': categorize_datasets_by_project}
        except ImportError:
            print("⚠️ Warning: categorization_logic module not found")
            modules['categorize'] = {}

        # Try to import enhanced_workflow module
        try:
            # print("DEBUG: Attempting to import create_enhanced_ai_workflow...")
            # from SRAgent.db.export import create_enhanced_ai_workflow
            # modules['enhanced_workflow'] = {'create_enhanced_ai_workflow': create_enhanced_ai_workflow}
            # print("DEBUG: create_enhanced_ai_workflow imported successfully into MODULES.")
            # Delay import of create_enhanced_ai_workflow until it's actually needed    
            modules['enhanced_workflow'] = {}
        except ImportError as e:
            print(f"⚠️ Warning: enhanced_workflow module not found: {e}")
            modules['enhanced_workflow'] = {}
            
    except Exception as e:
        print(f"⚠️ Warning: Could not import export modules: {e}")
        modules['enhanced_metadata'] = {}
        modules['categorize'] = {}
    
    print("DEBUG: sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    return modules

# Import modules
MODULES = import_required_modules()
if MODULES is None:
    print("❌ Critical modules missing. Exiting.")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Delay the import of enhanced-workflow, only loading when needed, to avoid circular dependency 
if not MODULES['enhanced_workflow']:
    try:
        from SRAgent.db.export import create_enhanced_ai_workflow       # noqa: E402
        MODULES['enhanced_workflow']['create_enhanced_ai_workflow'] = create_enhanced_ai_workflow
    except ImportError:
        pass

def run_enhanced_workflow(
    db_path: str = 'SRAgent.db',
    output_dir: str = 'enhanced_workflow_output',
    enable_categorization: bool = True,
    organisms: List[str] = ['human'],
    search_term: Optional[str] = None,
    limit: int = 10,
    min_sc_confidence: int = 2
) -> Dict[str, Any]:
    """
    Runs the enhanced AI workflow directly from get.py.

    Args:
        db_path: Path to the SQLite database.
        output_dir: Directory to save the output files.
        enable_categorization: Whether to enable categorization.
        organisms: List[str] = ['human'],
        search_term: Search term to filter datasets.
        limit: Maximum number of records to process.
        min_sc_confidence: Minimum single-cell confidence score.

    Returns:
        A dictionary containing the workflow execution status and results.
    """
    logger.info("Running enhanced AI workflow directly from get.py...")

    # Import db_connect with proper error handling
    db_connect = None
    try:
        from SRAgent.db.connect import db_connect
    except ImportError:
        try:
            # Try alternative import path if direct import fails
            current_dir = os.path.dirname(os.path.abspath(__file__))
            connect_path = os.path.join(current_dir, 'connect.py')
            if os.path.exists(connect_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("connect", connect_path)
                connect_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(connect_module)
                db_connect = connect_module.db_connect
            else:
                print("❌ Cannot import db_connect. Please check your setup.")
                return {"status": "error", "message": "Database connection module not found"}
        except Exception as e:
            print(f"❌ Cannot import db_connect: {e}")
            return {"status": "error", "message": f"Database connection module import failed: {e}"}

    if db_connect is None:
        return {"status": "error", "message": "Database connection function not available"}

    if 'get_prefiltered_datasets_functional' not in MODULES.get('prefilter_functions', {}):
        logger.error("get_prefiltered_datasets_functional not available. Please check imports.")
        return {"status": "error", "message": "Workflow function not found"}

    try:
        with db_connect() as conn:
            result = MODULES['prefilter_functions']['get_prefiltered_datasets_functional'](
                conn=conn,
                organisms=organisms,
                search_term=search_term,
                limit=limit,
                min_sc_confidence=min_sc_confidence,
                modules=MODULES # Pass the MODULES dictionary
            )
        logger.info(f"Workflow execution complete.")

        # for ai enhancement  
        ai_data = {}
        if enable_categorization and MODULES.get("categorize"):
            categorized = MODULES["categorize"]["categorize_datasets_by_project"](result)

            if MODULES.get("enhanced_metadata"):
                extractor_cls = MODULES["enhanced_metadata"]["EnhancedMetadataExtractor"]
                extractor = extractor_cls()
                ai_data = extractor.extract_hierarchical_metadata_from_db(conn, categorized)

        # Assuming result is a DataFrame from get_prefiltered_datasets_functional
        # Convert it to a dictionary format expected by the test workflow
        return {
            "status": "success" if not result.empty else "no_data",
            "output_directory": None, # This function doesn't create files directly
            "total_records": len(result) if not result.empty else 0,
            "total_experiments": len(result['experiment_id'].unique()) if 'experiment_id' in result.columns and not result.empty else 0,
            "categories": ai_data.get("category_summary") if ai_data else None, 
            "ai_data": ai_data, 
            "files_created": [] 
        }
    except Exception as e:
        logger.error(f"Error running enhanced workflow: {e}")
        return {"status": "error", "message": str(e)}

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

    if not MODULES.get('prefilter'):
        logger.error("prefilter module not available")
        return pd.DataFrame()
    
    # Map filter names to classes
    filter_map = MODULES['prefilter']['filters']
    
    try:
        # Build custom filter chain
        filter_chain = []
        result = None
        
        for filter_name in custom_filters:
            filter_class_name = f"{filter_name.title()}Filter"
            if filter_class_name not in filter_map:
                logger.warning(f"Unknown filter: {filter_name}")
                continue
            
            filter_class = filter_map[filter_class_name]
            
            # Create instance based on filter type
            if filter_name == 'organism':
                filter_obj = filter_class(conn, filter_params.get('organisms', ['human']))
            elif filter_name == 'single_cell':
                filter_obj = filter_class(conn, filter_params.get('min_sc_confidence', 2))
            elif filter_name == 'keyword':
                filter_obj = filter_class(conn, filter_params.get('search_term'))
            elif filter_name == 'limit':
                filter_obj = filter_class(conn, filter_params.get('limit', 20000000))
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
    limit: int = 20000000
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

def test_enhanced_workflow():
    """
    Test the enhanced workflow with better error handling
    """
    try:
        # Import db_connect with proper error handling
        try:
            from connect import db_connect
        except ImportError:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                sys.path.insert(0, os.path.join(project_root, 'SRAgent', 'db'))
                from connect import db_connect
            except ImportError:
                print("❌ Cannot import db_connect. Please check your setup.")
                return
        
        print("=== Testing Enhanced Workflow ===")

        # Test with small dataset  
        result = run_enhanced_workflow(
            organisms=["human"],
            search_term="cancer",
            limit=50
        )

        if result["status"] == "success":
            print(f"✅ Enhanced workflow successful!")
            if "data" in result:
                print(f"📊 Standard data: {len(result['data'])} records")
            if "ai_data" in result and result['ai_data']:
                hierarchical_data = result['ai_data'].get('hierarchical_data', {})
                if hierarchical_data:
                    total_experiments = sum(
                        data.get('category_summary', {}).get('total_experiments', 0) 
                        for data in hierarchical_data.values()
                    )
                    print(f"🤖 AI-optimized data: {total_experiments} experiments")
                        
                    # Show structure
                    for category, data in hierarchical_data.items():
                        if data.get('experiments'):
                            exp_count = len(data['experiments'])
                            print(f"  - {category}: {exp_count} experiments")
                else:
                    print("🤖 AI-optimized data: No hierarchical data generated")
            else:
                print("🤖 AI-optimized data: Not available (missing modules)")
        else:
            print(f"❌ Enhanced workflow failed: {result.get('message')}")
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


# main
if __name__ == "__main__":
    get_enhanced_prefiltered_datasets = run_enhanced_workflow
    from dotenv import load_dotenv
    load_dotenv()
    
    os.environ["DYNACONF"] = "test"  

    # Run tests
    print("🧪 Running module tests...")
    print(f"📦 Available modules: {list(MODULES.keys())}")

    # # Test basic functionality first
    # test_basic_prefiltering()
    
    # Test enhanced functionality if modules are available
    if MODULES.get('enhanced_metadata') and MODULES.get('categorize'):
        test_enhanced_workflow()
    else:
        print("\n⚠️ Enhanced workflow skipped - missing required modules:")
        if not MODULES.get('enhanced_metadata'):
            print("   - enhanced_metadata module")
        if not MODULES.get('categorize'):
            print("   - categorize module")
    
    print("\n🎯 Tests completed!")