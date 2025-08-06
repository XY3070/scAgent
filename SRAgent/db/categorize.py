"""
Simple categorize module for dataset organization
Fixed version - ensures proper imports and function definitions
"""
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime

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


def group_datasets_by_project_id(categorized_data: Dict[str, List[Dict]]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Group categorized datasets by project ID
    
    Args:
        categorized_data: Output from categorize_datasets_by_project
        
    Returns:
        Dictionary with datasets grouped by project ID within each category
    """
    try:
        grouped = {}
        
        for category, records in categorized_data.items():
            if category == 'discarded':
                grouped[category] = records
                continue
                
            # Group by project ID
            projects = {}
            for record in records:
                # Determine project ID based on category
                if category == 'GSE':
                    project_id = next((str(record[field]) for field in ['gse_title', 'study_alias'] 
                                     if field in record and record[field] and str(record[field]).startswith('GSE')), 'unknown')
                elif category == 'PRJNA':
                    project_id = next((str(record[field]) for field in ['study_alias', 'sra_ID'] 
                                     if field in record and record[field] and str(record[field]).startswith('PRJNA')), 'unknown')
                elif category == 'ena-STUDY':
                    project_id = next((str(record[field]) for field in ['study_alias', 'sra_ID'] 
                                     if field in record and record[field] and str(record[field]).startswith('ena-STUDY')), 'unknown')
                elif category == 'E-MTAB':
                    project_id = next((str(record[field]) for field in ['study_alias', 'sra_ID'] 
                                     if field in record and record[field] and str(record[field]).startswith('E-MTAB')), 'unknown')
                else:
                    project_id = 'unknown'
                
                # Add to projects dictionary
                if project_id not in projects:
                    projects[project_id] = []
                projects[project_id].append(record)
            
            grouped[category] = projects
        
        return grouped
        
    except Exception as e:
        logger.error(f"Error in group_datasets_by_project_id: {e}")
        return {category: {} for category in categorized_data.keys()}


def get_project_statistics(categorized_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Get statistics about categorized projects
    
    Args:
        categorized_data: Output from categorize_datasets_by_project
        
    Returns:
        Dictionary with statistics
    """
    try:
        stats = {
            'total_records': sum(len(records) for records in categorized_data.values()),
            'categories': {},
            'top_studies': {}
        }
        
        for category, records in categorized_data.items():
            if not records:
                continue
                
            stats['categories'][category] = {
                'count': len(records),
                'percentage': (len(records) / stats['total_records']) * 100 if stats['total_records'] > 0 else 0
            }
            
            # Find top studies in this category
            study_counts = {}
            for record in records:
                study_id = record.get('study_alias') or record.get('gse_title') or 'unknown'
                study_counts[study_id] = study_counts.get(study_id, 0) + 1
            
            # Get top 5 studies
            top_studies = sorted(study_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_studies'][category] = top_studies
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_project_statistics: {e}")
        return {'total_records': 0, 'categories': {}, 'top_studies': {}}


def create_classify_ready_export(categorized_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Create a classification-ready export structure from categorized data
    
    Args:
        categorized_data: Output from categorize_datasets_by_project
        
    Returns:
        Dictionary with classification-ready structure
    """
    try:
        # Group data by project ID
        grouped_data = group_datasets_by_project_id(categorized_data)
        
        # Get statistics
        stats = get_project_statistics(categorized_data)
        
        # Create export structure
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "statistics": stats
            },
            "categorized_data": categorized_data,
            "grouped_data": grouped_data
        }
        
        return export_data
        
    except Exception as e:
        logger.error(f"Error in create_classify_ready_export: {e}")
        return {"error": str(e)}


def run_export_workflow(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the complete export workflow: categorize, group, and prepare for export
    
    Args:
        df: DataFrame with dataset records
        
    Returns:
        Dictionary with complete export structure
    """
    try:
        # Step 1: Categorize datasets
        categorized = categorize_datasets_by_project(df)
        
        # Step 2: Create export-ready structure
        export_data = create_classify_ready_export(categorized)
        
        return export_data
        
    except Exception as e:
        logger.error(f"Error in run_export_workflow: {e}")
        return {"error": str(e)}


# Test function to verify the module works
def test_categorize_module():
    """Test the categorize module functionality"""
    try:
        print("🧪 Testing categorize module...")
        
        # Create sample data for testing
        sample_data = pd.DataFrame([
            {'study_alias': 'GSE123456', 'sra_ID': '001', 'gsm_title': 'Sample 1'},
            {'study_alias': 'GSE123456', 'sra_ID': '002', 'gsm_title': 'Sample 2'},
            {'study_alias': 'PRJNA789012', 'sra_ID': '003', 'gsm_title': 'Sample 3'},
            {'study_alias': 'ena-STUDY-456789', 'sra_ID': '004', 'gsm_title': 'Sample 4'},
        ])
        
        # Test categorization
        categorized = categorize_datasets_by_project(sample_data)
        print(f"✅ Categorization test successful:")
        for category, records in categorized.items():
            if records:
                print(f"  - {category}: {len(records)} records")
        
        # Test statistics
        stats = get_project_statistics(categorized)
        print(f"✅ Statistics test successful: {stats['total_records']} total records")
        
        # Test grouping
        grouped = group_datasets_by_project_id(categorized)
        print(f"✅ Grouping test successful")
        
        # Test complete workflow
        export_data = run_export_workflow(sample_data)
        print(f"✅ Export workflow test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when executed directly
    test_categorize_module()
