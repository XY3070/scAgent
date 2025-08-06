from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def categorize_datasets_by_project(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Categorize datasets by project type (GSE, PRJNA, etc.)
    
    Args:
        df: DataFrame with dataset records
        
    Returns:
        Dictionary with categorized records
    """
    try:
        if df.empty:
            logger.info("Empty DataFrame provided, returning empty categories")
            return {'GSE': [], 'PRJNA': [], 'ena-STUDY': [], 'discarded': []}
        
        categorized = {'GSE': [], 'PRJNA': [], 'ena-STUDY': [], 'E-MTAB': [],'discarded': []}
        records = df.to_dict(orient='records')
        
        logger.info(f"Categorizing {len(records)} records")
        
        for record in records:
            project_id = None
            
            # Search fields for project identifiers
            search_fields = [
                'study_alias', 'sra_ID', 'run_alias', 'experiment_alias', 
                'sample_alias', 'submission_alias', 'gsm_title', 'gse_title'
            ]
            
            for field in search_fields:
                if field in record and record[field]:
                    value = str(record[field]).strip()
                    if value and value not in ['nan', 'None', '', 'null']:
                        project_id = value
                        break
            
            if not project_id:
                categorized['discarded'].append(record)
                continue
                
            # Categorize based on project ID prefix
            if project_id.startswith('GSE'):
                categorized['GSE'].append(record)
            elif project_id.startswith('PRJNA'):
                categorized['PRJNA'].append(record)
            elif project_id.startswith('ena-STUDY'):
                categorized['ena-STUDY'].append(record)
            elif project_id.startswith('E-MTAB'):
                categorized['E-MTAB'].append(record)
            else:
                # Try to extract from other fields
                found_category = False
                for field_name, field_value in record.items():
                    if field_value and isinstance(field_value, str):
                        field_str = str(field_value).upper()
                        if 'GSE' in field_str:
                            categorized['GSE'].append(record)
                            found_category = True
                            break
                        elif 'PRJNA' in field_str:
                            categorized['PRJNA'].append(record)
                            found_category = True
                            break
                        elif 'ena-STUDY' in field_str:
                            categorized['ena-STUDY'].append(record)
                            found_category = True
                            break
                        elif 'E-MTAB' in field_str:
                            categorized['E-MTAB'].append(record)
                            found_category = True
                            break
                
                if not found_category:
                    categorized['discarded'].append(record)
        
        # Log categorization results
        total_categorized = sum(len(records) for category, records in categorized.items() if category != 'discarded')
        logger.info(f"Categorization complete:")
        for category, records in categorized.items():
            if records:
                logger.info(f"  - {category}: {len(records)} records")
        
        logger.info(f"Total categorized: {total_categorized}, Discarded: {len(categorized['discarded'])}")
        
        return categorized
        
    except Exception as e:
        logger.error(f"Error in categorize_datasets_by_project: {e}")
        # Return empty structure on error
        return {'GSE': [], 'PRJNA': [], 'ena-STUDY': [], 'E-MTAB': [], 'discarded': []}

def group_datasets_by_project_id(categorized_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Groups datasets by project ID within each category.

    Args:
        categorized_data (Dict[str, Any]): Categorized datasets.

    Returns:
        Dict[str, Any]: A dictionary with datasets grouped by project ID within each category.
    """
    grouped_data = {}
    for category, datasets in categorized_data.items():
        if category == 'discarded':
            continue
        grouped_data[category] = {}
        for dataset in datasets:
            project_id = dataset.get('project_id')
            if project_id:
                if project_id not in grouped_data[category]:
                    grouped_data[category][project_id] = []
                grouped_data[category][project_id].append(dataset)
    return grouped_data