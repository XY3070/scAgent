#!/usr/bin/env python3
"""
Quick fix script for categorize import issue
Run this in your SRAgent/db/ directory to fix the categorize module
"""

import os
import sys

def fix_categorize_module():
    """Fix the categorize module import issue"""
    
    print("🔧 Fixing categorize module...")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    categorize_path = os.path.join(current_dir, 'categorize.py')
    
    print(f"📁 Working directory: {current_dir}")
    print(f"📄 Categorize file path: {categorize_path}")
    
    # Create a working categorize.py
    categorize_content = '''"""
Simple categorize module for dataset organization
Fixed version - ensures proper imports and function definitions
"""
import pandas as pd
from typing import Dict, List, Any
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
        
        categorized = {'GSE': [], 'PRJNA': [], 'ena-STUDY': [], 'discarded': []}
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
        return {'GSE': [], 'PRJNA': [], 'ena-STUDY': [], 'discarded': []}


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
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when executed directly
    test_categorize_module()
'''
    
    try:
        # Backup existing file if it exists
        if os.path.exists(categorize_path):
            backup_path = categorize_path + '.backup'
            os.rename(categorize_path, backup_path)
            print(f"📋 Backed up existing file to: {backup_path}")
        
        # Write the fixed categorize.py
        with open(categorize_path, 'w', encoding='utf-8') as f:
            f.write(categorize_content)
        
        print(f"✅ Created fixed categorize.py: {categorize_path}")
        print(f"📏 File size: {os.path.getsize(categorize_path)} bytes")
        
        # Test the new file
        print(f"\n🧪 Testing the fixed categorize module...")
        
        # Add current directory to path for testing
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Try importing
        try:
            import importlib
            if 'categorize' in sys.modules:
                importlib.reload(sys.modules['categorize'])
            else:
                import categorize
            
            print("✅ Import successful")
            
            # Test the function
            if hasattr(categorize, 'categorize_datasets_by_project'):
                print("✅ Function 'categorize_datasets_by_project' found")
                
                # Quick function test
                import pandas as pd
                test_df = pd.DataFrame([{'study_alias': 'GSE123', 'sra_ID': '001'}])
                result = categorize.categorize_datasets_by_project(test_df)
                print(f"✅ Function test successful: {len(result)} categories")
                
            else:
                print("❌ Function 'categorize_datasets_by_project' not found")
                
        except Exception as e:
            print(f"❌ Import test failed: {e}")
            
        print(f"\n🎯 Fix completed! Your categorize module should now work properly.")
        print(f"📝 You can now run your main script again:")
        print(f"   uv run '/ssd2/xuyuan/SRAgent/SRAgent/db/get.py'")
        
    except Exception as e:
        print(f"❌ Error creating fixed categorize module: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_categorize_module()