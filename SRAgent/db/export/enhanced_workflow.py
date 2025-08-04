# Integration example for existing workflow
# File: SRAgent/db/export/enhanced_workflow.py

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import existing modules
from ..get import get_prefiltered_datasets_functional
from .categorize import categorize_datasets_by_project
from .enhanced_metadata import EnhancedMetadataExtractor, LocalPostgreSQLAdapter
from ..connect import db_connect

def create_enhanced_ai_workflow(
    conn,
    organisms: List[str] = ["human"],
    search_term: str = None,
    limit: int = 1000,
    output_dir: str = "enhanced_export"
) -> Dict[str, Any]:
    """
    Complete enhanced workflow for AI-optimized metadata extraction
    
    This integrates with your existing pipeline:
    1. Use existing prefilter functions
    2. Apply existing categorization
    3. Add enhanced AI-optimized structure
    4. Generate multiple output formats
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Step 1: Prefiltering datasets...")
    # Use your existing prefilter function
    df = get_prefiltered_datasets_functional(
        conn=conn,
        organisms=organisms,
        search_term=search_term,
        limit=limit
    )
    
    if df.empty:
        return {"status": "no_data", "message": "No records found after prefiltering"}
    
    print(f"✅ Found {len(df)} records after prefiltering")
    
    print("🔍 Step 2: Categorizing by project type...")
    # Use existing categorization (with your fixed version)
    categorized_data = categorize_datasets_by_project(df)
    
    print("📊 Categorization results:")
    for category, records in categorized_data.items():
        print(f"  - {category}: {len(records)} records")
    
    print("🔍 Step 3: Enhanced AI-optimized extraction...")
    # Apply enhanced extraction
    extractor = EnhancedMetadataExtractor()
    enhanced_data = extractor.extract_hierarchical_metadata_from_db(conn, categorized_data)
    
    print("🔍 Step 4: Generating outputs...")
    
    # Output 1: Full enhanced export (for AI agent)
    ai_export_path = output_dir / "ai_optimized_export.json"
    with open(ai_export_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, default=str, ensure_ascii=False)
    
    # Output 2: Structured metadata only (for quick AI processing)
    structured_data = {}
    total_experiments = 0
    
    for category, data in enhanced_data.get('hierarchical_data', {}).items():
        structured_data[category] = {}
        experiments = data.get('experiments', {})
        total_experiments += len(experiments)
        
        for exp_id, exp_data in experiments.items():
            structured_data[category][exp_id] = {
                'shared_metadata': exp_data.get('shared_metadata', {}),
                'ai_targeted_metadata': exp_data.get('ai_targeted_metadata', {}),
                'summary': exp_data.get('experiment_summary', {})
            }
    
    structured_path = output_dir / "structured_for_ai.json"
    with open(structured_path, 'w', encoding='utf-8') as f:
        json.dump({
            "extraction_info": enhanced_data.get('ai_processing_metadata', {}),
            "structured_experiments": structured_data
        }, f, indent=2, default=str, ensure_ascii=False)
    
    # Output 3: AI Processing Instructions
    instructions = create_ai_processing_instructions(enhanced_data)
    instructions_path = output_dir / "ai_processing_instructions.md"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    # Output 4: Category-specific files (like your original approach)
    for category, data in enhanced_data.get('hierarchical_data', {}).items():
        if data.get('experiments'):
            category_path = output_dir / f"{category.lower()}_enhanced.json"
            with open(category_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "category": category,
                    "category_summary": data.get('category_summary', {}),
                    "experiments": data.get('experiments', {})
                }, f, indent=2, default=str, ensure_ascii=False)
    
    print("✅ Enhanced workflow completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total experiments processed: {total_experiments}")
    
    return {
        "status": "success",
        "output_directory": str(output_dir),
        "total_experiments": total_experiments,
        "total_records": len(df),
        "categories": list(categorized_data.keys()),
        "files_created": [
            str(ai_export_path),
            str(structured_path),
            str(instructions_path)
        ]
    }

def create_ai_processing_instructions(enhanced_data: Dict[str, Any]) -> str:
    """
    Create AI processing instructions based on the enhanced data structure
    """
    
    instructions = f"""# AI Processing Instructions
Generated: {datetime.now().isoformat()}

## Data Structure Overview

This export contains hierarchically structured metadata optimized for AI processing:

### 1. Experiment-Level Processing
For each experiment, process in this order:
1. **Shared Metadata**: Information common to all runs in the experiment
2. **AI-Targeted Metadata**: Pre-structured data for specific extraction tasks
3. **Individual Records**: Run-specific information

### 2. Required Extractions

#### High Priority (Must Extract):
- **tumor_status**: Boolean classification (tumor=True/False)
- **sequencing_method**: Technology and platform information
- **tissue_source**: Anatomical location and tissue type
- **data_access**: URLs and accession numbers for raw data

#### Medium Priority:
- **publication_info**: PMID, DOI, or paper links
- **sample_count**: Number of samples and data volume
- **geographic_info**: Sample origin country/region

#### Low Priority:
- **age_info**: Age ranges or specific values (if available)

### 3. Processing Strategy

```python
# Example processing logic:
for category in ["GSE", "PRJNA", "ENA"]:
    for experiment_id, experiment_data in data[category]["experiments"].items():
        
        # Step 1: Extract shared context (process once per experiment)
        shared_context = experiment_data["shared_metadata"]
        
        # Step 2: Use AI-targeted metadata for structured extraction
        ai_metadata = experiment_data["ai_targeted_metadata"]
        
        # Step 3: Process individual records
        for record in experiment_data["individual_records"]:
            # Extract record-specific information
            pass
```

### 4. Field Mapping Guide

| Required Field | Primary Sources | Fallback Sources |
|----------------|----------------|------------------|
| tumor_status | characteristics_ch1 | source_name_ch1, gsm_description |
| sequencing_method | library_strategy | platform, instrument_model |
| tissue_source | characteristics_ch1 | source_name_ch1, organism_ch1 |
| data_access | study_xref_link | gse_supplementary_file |
| publication_info | pubmed_id | study_url_link, gse_web_link |

### 5. Output Format

Generate structured JSON output with this schema:
```json
{{
  "experiment_id": "GSE123456",
  "extracted_metadata": {{
    "tumor_status": true/false,
    "sequencing_method": "scRNA-seq",
    "tissue_source": "brain tissue",
    "data_access": ["ftp://...", "http://..."],
    "publication_info": {{"pmid": "12345", "doi": "10.1000/..."}},
    "sample_count": 1000,
    "geographic_info": "USA",
    "age_info": "adult"
  }},
  "individual_runs": [
    {{
      "sra_ID": "12345.0",
      "run_specific_metadata": {{...}}
    }}
  ]
}}
```

### 6. Error Handling

- If required fields are missing, mark as "not_available"
- If conflicting information exists, prefer more recent/detailed sources
- Log extraction confidence levels for quality control

"""
    
    # Add category-specific statistics
    hierarchical_data = enhanced_data.get('hierarchical_data', {})
    if hierarchical_data:
        instructions += "\n### 7. Data Statistics\n\n"
        for category, data in hierarchical_data.items():
            summary = data.get('category_summary', {})
            instructions += f"**{category}**:\n"
            instructions += f"- Experiments: {summary.get('total_experiments', 0)}\n"
            instructions += f"- Records: {summary.get('total_records', 0)}\n"
            instructions += f"- Avg records/experiment: {summary.get('avg_records_per_experiment', 0):.1f}\n\n"
    
    return instructions

# Integration with existing get.py
def add_to_existing_get_py():
    """
    Code to add to your existing get.py file for integration
    """
    integration_code = '''
# Add this to your existing get.py file

def get_enhanced_prefiltered_datasets(
    conn: connection,
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 20000000,
    min_sc_confidence: int = 2,
    output_format: str = "ai_optimized"  # "ai_optimized", "standard", "both"
) -> Dict[str, Any]:
    """
    Enhanced version of prefiltering with AI-optimized output
    
    Returns both standard DataFrame and AI-optimized hierarchical structure
    """
    try:
        # Use existing prefilter function
        result_df = get_prefiltered_datasets_functional(
            conn=conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit,
            min_sc_confidence=min_sc_confidence
        )
        
        if result_df.empty:
            return {"status": "no_data", "data": pd.DataFrame(), "ai_data": {}}
        
        # Apply categorization and enhancement
        from .export.categorize import categorize_datasets_by_project
        from .export.enhanced_metadata import EnhancedMetadataExtractor
        
        categorized = categorize_datasets_by_project(result_df)
        
        if output_format in ["ai_optimized", "both"]:
            extractor = EnhancedMetadataExtractor()
            ai_data = extractor.extract_hierarchical_metadata_from_db(conn, categorized)
        else:
            ai_data = {}
        
        if output_format == "ai_optimized":
            return {"status": "success", "ai_data": ai_data}
        elif output_format == "both":
            return {"status": "success", "data": result_df, "ai_data": ai_data}
        else:
            return {"status": "success", "data": result_df}
            
    except Exception as e:
        logger.error(f"Enhanced prefiltering failed: {e}")
        return {"status": "error", "message": str(e)}
'''
    return integration_code

# CLI interface example
def main():
    """
    Example CLI usage of the enhanced workflow
    """
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Enhanced AI-optimized metadata extraction')
    parser.add_argument('--organisms', nargs='+', default=['human'], help='List of organisms')
    parser.add_argument('--search-term', default='cancer', help='Search term')
    parser.add_argument('--limit', type=int, default=100, help='Maximum records')
    parser.add_argument('--output-dir', default='enhanced_export', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        with db_connect() as conn:
            result = create_enhanced_ai_workflow(
                conn=conn,
                organisms=args.organisms,
                search_term=args.search_term,
                limit=args.limit,
                output_dir=args.output_dir
            )
            
            if result["status"] == "success":
                print(f"✅ Success! Created {len(result['files_created'])} output files")
                print(f"📊 Processed {result['total_experiments']} experiments")
            else:
                print(f"❌ Failed: {result.get('message', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()