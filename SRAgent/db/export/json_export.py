import json
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import json
from psycopg2.extensions import connection

from ..get import get_prefiltered_datasets_functional
from SRAgent.db.categorization_logic import categorize_datasets_by_project, group_datasets_by_project_id

logger = logging.getLogger(__name__)


def export_prefiltered_to_json(
    conn: connection,
    output_path: str = "prefiltered_datasets.json",
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 100000,
    min_sc_confidence: int = 2,
    include_categorization: bool = True,
    include_grouping: bool = True
) -> Dict[str, Any]:
    """
    Export prefiltered datasets to JSON file with optional categorization and grouping
    
    Args:
        conn: Database connection object
        output_path: Path to save JSON file
        organisms: List of organisms to filter
        search_term: Search term to filter datasets
        limit: Maximum number of records
        min_sc_confidence: Minimum single-cell confidence score
        include_categorization: Whether to categorize by project type
        include_grouping: Whether to group by project ID
    """
    try:
        # Get prefiltered datasets  
        logger.info("Getting prefiltered datasets...")
        df = get_prefiltered_datasets_functional(
            conn=conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit,
            min_sc_confidence=min_sc_confidence
        )

        if df.empty:
            logger.warning("No records found with current filters")
            return {"status": "no_data", "message": "No records found"}

        # Prepare to export data  
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_records": len(df),
                "filter_parameters": {
                    "organisms": organisms,
                    "search_term": search_term,
                    "limit": limit,
                    "min_sc_confidence": min_sc_confidence
                }
            },
            "raw_data": df.to_dict(orient='records')
        }

        # Add categorization if requested
        if include_categorization:
            logger.info("Categorizing datasets by project type...")
            categorized = categorize_datasets_by_project(df)
            export_data["categorized_data"] = categorized
            
            # Add categorization statistics
            export_data["export_metadata"]["categorization_stats"] = {
                category: len(records) for category, records in categorized.items()
            }
        
        # Add grouping if requested
        if include_grouping and include_categorization:
            logger.info("Grouping datasets by project ID...")
            grouped = group_datasets_by_project_id(categorized)
            export_data["grouped_data"] = grouped
            
            # Add grouping statistics
            grouping_stats = {}
            for category, projects in grouped.items():
                if category == 'discarded':
                    continue
                grouping_stats[category] = {
                    "unique_projects": len(projects),
                    "projects": {pid: len(datasets) for pid, datasets in projects.items()}
                }
            export_data["export_metadata"]["grouping_stats"] = grouping_stats
        
        # Write to JSON file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Successfully exported {len(df)} records to {output_path}")
        
        return {
            "status": "success",
            "output_path": str(output_path),
            "total_records": len(df),
            "export_data": export_data["export_metadata"]
        }
        
    except Exception as e:
        logger.error(f"Failed to export to JSON: {e}")
        return {"status": "error", "message": str(e)}

def export_ai_data_to_json(
    ai_data: Dict[str, Any],
    output_directory: str = "./output",
    filename: str = "ai_enhanced_data.json"
) -> Dict[str, Any]:
    """
    Export AI enhanced data to a JSON file.

    Args:
        ai_data: The dictionary containing AI enhanced data.
        output_directory: The directory where the JSON file will be saved.
        filename: The name of the JSON file.

    Returns:
        A dictionary indicating the status of the export operation.
    """
    try:
        output_path = Path(output_directory) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ai_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Successfully exported AI enhanced data to {output_path}")
        return {"status": "success", "output_path": str(output_path)}
    except Exception as e:
        logger.error(f"Failed to export AI enhanced data to JSON: {e}")
        return {"status": "error", "message": str(e)}
