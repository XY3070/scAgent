import sys
import os
from pathlib import Path

# Add the project root to the sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import necessary functions
from SRAgent.db.connect import db_connect
from SRAgent.db.get import run_enhanced_workflow

def run_test_workflow():
    conn = None
    try:
        print("Attempting to connect to the database...")
        conn = db_connect()
        if conn:
            print("Database connection successful.")
            
            output_dir = "test_output_categorization"
            print(f"Running enhanced AI workflow, output will be in: {output_dir}")
            
            # Run the workflow with categorization enabled
            results = run_enhanced_workflow(
                organisms=["human"],
                limit=10, # Limit to a small number for quick testing


                min_sc_confidence=2
            )
            
            print("Workflow execution complete.")
            print(f"Status: {results.get('status')}")
            print(f"Output Directory: {results.get('output_directory')}")
            print(f"Total Records Processed: {results.get('total_records')}")
            print(f"Total Experiments Processed: {results.get('total_experiments')}")
            print(f"Categories Found: {results.get('categories')}")
            print(f"Files Created: {results.get('files_created')}")
            
            # Verify output files exist
            for f_path in results.get('files_created', []):
                if Path(f_path).exists():
                    print(f"  ✅ File exists: {f_path}")
                    # Optionally, load and inspect the main JSON output
                    if 'ai_optimized_export.json' in f_path:
                        with open(f_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            print(f"    Loaded {len(data.get('hierarchical_data', {}))} categories from ai_optimized_export.json")
                            # Further checks can be added here, e.g., check for 'GSE' or 'PRJNA' keys
                            if 'hierarchical_data' in data:
                                for category, cat_data in data['hierarchical_data'].items():
                                    print(f"      Category '{category}': {len(cat_data.get('experiments', {}))} experiments")
                else:
                    print(f"  ❌ File NOT found: {f_path}")

        else:
            print("Failed to establish database connection.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    import json # Import json here for use in __main__ block
    run_test_workflow()