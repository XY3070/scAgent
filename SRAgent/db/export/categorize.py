import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

from .json_export import export_prefiltered_datasets_to_json
from .db_export import export_prefiltered_datasets_to_sqlite, export_prefiltered_datasets_to_postgres
from ..connect import db_connect

logger = logging.getLogger(__name__)


def categorize_datasets_by_project(df) -> Dict[str, List[Dict]]:
    """
    Categorize datasets by project prefix (GSE< PRJNA, ENA)

    Args:
        df: DataFrame containing prefiltered datasets  

    Returns:
        Dictionary with project types as keys and grouped datasets as values 
    """
    catogrized = {
        'GSE': [],
        'PRJNA': [],
        'ENA': []
    }

    # Convert DataFrame to list of dictionaries for easier processng 
    records = df.to_dict(orient='records')

    for record in records:
        # Extract project identifier from various potential fields  
        project_id = None
        
        # Check stufy_alias for project identifiers 
        for field in  ['study_alias', 'sample_alias']:
            if field in record and record[field]:
                project_id = str(record[field])
                break
        
        # Categorize based on project prefix
        if project_id:
            if project_id.startswith('GSE'):
                catogrized['GSE'].append(record)
            elif project_id.startswith('PRJNA'):
                catogrized['PRJNA'].append(record)
            elif project_id.startswith('ena-STUDY'):
                catogrized['ENA'].append(record)
    return catogrized

def group_datasets_by_project_id(catogorized: Dict[str, List[Dict]]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Group datasets by project ID within each category.
    
    Args:
        categorized: output from categorize_datasets_by_project 
        
    Returns:
        Nested dictionary: {category: {project_id: [datasets]}}
    """
    grouped = {}

    for category, records in categorized.items():
        if category == 'discarded':
            grouped[category] = records
            continue
            
        grouped[category] = {}
        
        for record in records:
            # Extract project ID
            project_id = None
            for field in ['study_alias', 'sample_alias']:
                if field in record and record[field]:
                    project_id = str(record[field])
                    break
            
            if project_id:
                if project_id not in grouped[category]:
                    grouped[category][project_id] = []
                grouped[category][project_id].append(record)
    
    return grouped

def create_classify_ready_export(
    conn: connection,
    output_dir: str = "classify_export",
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 1000,
    min_sc_confidence: int = 2,
    export_format: str = "json"  # "json", "sqlite", or "postgres"
) -> Dict[str, Any]:
    """
    Create export specifically optimized for classify.py workflow
    
    Args:
        conn: Database connection
        output_dir: Directory for export files
        organisms: List of organisms to filter
        search_term: Search term for filtering
        limit: Maximum number of records
        min_sc_confidence: Minimum single-cell confidence
        export_format: Export format ("json", "sqlite", or "postgres")
    
    Returns:
        Dictionary containing export results and paths
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        if export_format == "json":
            # Export to JSON with full categorization and grouping
            json_path = output_dir / "classified_datasets.json"
            result = export_prefiltered_datasets_to_json(
                conn=conn,
                output_path=str(json_path),
                organisms=organisms,
                search_term=search_term,
                limit=limit,
                min_sc_confidence=min_sc_confidence,
                include_categorization=True,
                include_grouping=True
            )
            results["json_export"] = result
            
            # Create separate JSON files for each project category for easier processing
            if result["status"] == "success":
                with open(json_path, 'r') as f:
                    export_data = json.load(f)
                
                if "grouped_data" in export_data:
                    for category, projects in export_data["grouped_data"].items():
                        if category == "discarded" or not projects:
                            continue
                        
                        category_file = output_dir / f"{category.lower()}_projects.json"
                        category_data = {
                            "category": category,
                            "export_metadata": export_data["export_metadata"],
                            "projects": projects
                        }
                        
                        with open(category_file, 'w') as f:
                            json.dump(category_data, f, indent=2, default=str)
                        
                        results[f"{category.lower()}_file"] = str(category_file)
        
        elif export_format == "sqlite":
            # Export to SQLite
            db_path = output_dir / "classified_datasets.db"
            result = export_prefiltered_datasets_to_sqlite(
                conn=conn,
                output_path=str(db_path),
                organisms=organisms,
                search_term=search_term,
                limit=limit,
                min_sc_confidence=min_sc_confidence,
                create_categorized_tables=True
            )
            results["sqlite_export"] = result
        
        elif export_format == "postgres":
            # Export to PostgreSQL
            db_name = "classified_datasets"
            result = export_prefiltered_datasets_to_postgres(
                source_conn=conn,
                export_db_name=db_name,
                organisms=organisms,
                search_term=search_term,
                limit=limit,
                min_sc_confidence=min_sc_confidence,
                create_categorized_tables=True
            )
            results["postgres_export"] = result
        
        # Create a summary file for the classify.py workflow
        summary = {
            "export_summary": {
                "timestamp": datetime.now().isoformat(),
                "export_format": export_format,
                "output_directory": str(output_dir),
                "filter_parameters": {
                    "organisms": organisms,
                    "search_term": search_term,
                    "limit": limit,
                    "min_sc_confidence": min_sc_confidence
                },
                "results": results
            }
        }
        
        summary_path = output_dir / "export_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        results["summary_file"] = str(summary_path)
        
        logger.info(f"Created classify-ready export in {output_dir}")
        return {
            "status": "success",
            "output_directory": str(output_dir),
            "export_format": export_format,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to create classify-ready export: {e}")
        return {"status": "error", "message": str(e)}

def run_export_workflow(
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 1000,
    min_sc_confidence: int = 2,
    output_format: str = "json",
    output_path: str = None
):
    """
    Run the complete export workflow - a convenient function for CLI usage
    
    Args:
        organisms: List of organisms to filter
        search_term: Search term for filtering
        limit: Maximum number of records
        min_sc_confidence: Minimum single-cell confidence
        output_format: Export format ("json", "sqlite", "postgres")
        output_path: Custom output path (optional)
    """
    try:
        with db_connect() as conn:
            print(f"🔍 Starting export workflow with format: {output_format}")
            print(f"📊 Filter parameters:")
            print(f"  - Organisms: {organisms}")
            print(f"  - Search term: {search_term}")
            print(f"  - Limit: {limit}")
            print(f"  - Min SC confidence: {min_sc_confidence}")
            
            if output_path is None:
                output_path = f"exports/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = create_classify_ready_export(
                conn=conn,
                output_dir=output_path,
                organisms=organisms,
                search_term=search_term,
                limit=limit,
                min_sc_confidence=min_sc_confidence,
                export_format=output_format
            )
            
            if result["status"] == "success":
                print(f"✅ Export completed successfully!")
                print(f"📁 Output directory: {result['output_directory']}")
                print(f"📋 Results: {result['results']}")
            else:
                print(f"❌ Export failed: {result.get('message', 'Unknown error')}")
                
            return result
            
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return {"status": "error", "message": str(e)}
