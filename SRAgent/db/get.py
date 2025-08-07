import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
from psycopg2.extensions import connection
import logging
from datetime import datetime


# Fix import path issues
import os
import sys

# Add the parent directory of SRAgent to sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now import modules with proper error handling
def import_required_modules():
    """
    Import required modules with fallback strategies.
    """
    modules = {}
    
    # Try importing prefilter_functions module
    try:
        from SRAgent.db.prefilter_functions import get_prefiltered_datasets_functional
        modules['prefilter_functions'] = {'get_prefiltered_datasets_functional': get_prefiltered_datasets_functional}
    except ImportError as e:
        print(f"❌ Cannot import prefilter_functions module: {e}")
        print("Please ensure SRAgent/db/prefilter_functions.py is in your Python path or accessible.")
        return None

    # Try importing prefilter module
    try:
        from SRAgent.db.prefilter import (
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
        print("Please ensure SRAgent/db/prefilter.py is in your Python path or accessible.")
        return None
    
    # Try importing utils
    try:
        from SRAgent.db.utils import execute_query
        modules['utils'] = {'execute_query': execute_query}
    except ImportError as e:
        print(f"❌ Cannot import utils module: {e}")
        print("Please ensure SRAgent/db/utils.py is in your Python path or accessible.")
        return None
    
    # Try importing export modules
    try:
        # Try to import enhanced_metadata
        try:
            from SRAgent.db.enhanced_metadata import EnhancedMetadataExtractor, enhance_existing_categorize_workflow
            modules['enhanced_metadata'] = {
                'EnhancedMetadataExtractor': EnhancedMetadataExtractor,
                'enhance_existing_categorize_workflow': enhance_existing_categorize_workflow
            }
        except ImportError:
            print("⚠️ Warning: enhanced_metadata module not found")
            modules['enhanced_metadata'] = {}
        
        # Try to import categorization_logic module
        try:
            from SRAgent.db.categorization_logic import categorize_datasets_by_project

            modules['categorize'] = {'categorize_datasets_by_project': categorize_datasets_by_project}
        except ImportError:
            print("⚠️ Warning: categorization_logic module not found")
            modules['categorize'] = {}

        # Try to import enhanced_workflow module
        try:
            from SRAgent.db.export.json_export import export_ai_data_to_json as create_enhanced_ai_workflow
            modules['enhanced_workflow'] = {'create_enhanced_ai_workflow': create_enhanced_ai_workflow}
        except ImportError as e:
            print(f"⚠️ Warning: enhanced_workflow module not found: {e}")
            modules['enhanced_workflow'] = {}
            
    except Exception as e:
        print(f"⚠️ Warning: Could not import export modules: {e}")
        modules['enhanced_metadata'] = {}
        modules['categorize'] = {}
        modules['enhanced_workflow'] = {}

    print(f"DEBUG: __name__: {__name__}")
    print(f"DEBUG: __package__: {__package__}")
    print("DEBUG: sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    return modules

# Import modules
logger = logging.getLogger(__name__)

MODULES = import_required_modules()
if MODULES is None:
    logger.critical("❌ Critical modules missing. Exiting.")
    sys.exit(1)




def run_enhanced_workflow(
    db_path: str = 'SRAgent.db',
    output_dir: str = '/ssd2/xuyuan/output',
    enable_categorization: bool = True,
    organisms: List[str] = ['human'],
    search_term: Optional[str] = None,
    limit: int = 2000,
    min_sc_confidence: int = 2,
    export_json: bool = True
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

    files_created = [] # Initialize files_created list

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
                
                if export_json and MODULES.get('enhanced_workflow') and 'create_enhanced_ai_workflow' in MODULES['enhanced_workflow']:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_filename = f"ai_enhanced_data_{timestamp}.json"
                    export_result = MODULES['enhanced_workflow']['create_enhanced_ai_workflow'](
                        ai_data, 
                        output_dir, 
                        json_filename
                    )
                    if export_result["status"] == "success":
                        files_created.append(export_result["output_path"])

        # Assuming result is a DataFrame from get_prefiltered_datasets_functional
        # Convert it to a dictionary format expected by the test workflow
        return {
            "status": "success" if not result.empty else "no_data",
            "output_directory": output_dir if files_created else None, # Reflect actual output directory if files were created
            "total_records": len(result) if not result.empty else 0,
            "total_experiments": len(result['experiment_id'].unique()) if 'experiment_id' in result.columns and not result.empty else 0,
            "categories": ai_data.get("category_summary") if ai_data else None, 
            "ai_data": ai_data, 
            "files_created": files_created, # Return the list of created files
            "export_json_enabled": export_json,
            "enhanced_workflow_module_available": bool(MODULES.get('enhanced_workflow')),
            "create_enhanced_ai_workflow_function_available": 'create_enhanced_ai_workflow' in MODULES.get('enhanced_workflow', {})
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
            # elif filter_name == 'keyword':
            #     filter_obj = filter_class(conn, filter_params.get('search_term'))
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
            limit=2000,
            export_json=True # Explicitly enable JSON export for testing
        )

        print(f"Export JSON enabled: {result.get('export_json_enabled')}")
        print(f"Enhanced workflow module available: {result.get('enhanced_workflow_module_available')}")
        print(f"Create enhanced AI workflow function available: {result.get('create_enhanced_ai_workflow_function_available')}")

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
    