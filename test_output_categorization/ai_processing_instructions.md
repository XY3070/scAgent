# AI Processing Instructions
Generated: 2025-08-06T21:05:45.360704

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
{
  "experiment_id": "GSE123456",
  "extracted_metadata": {
    "tumor_status": true/false,
    "sequencing_method": "scRNA-seq",
    "tissue_source": "brain tissue",
    "data_access": ["ftp://...", "http://..."],
    "publication_info": {"pmid": "12345", "doi": "10.1000/..."},
    "sample_count": 1000,
    "geographic_info": "USA",
    "age_info": "adult"
  },
  "individual_runs": [
    {
      "sra_ID": "12345.0",
      "run_specific_metadata": {...}
    }
  ]
}
```

### 6. Error Handling

- If required fields are missing, mark as "not_available"
- If conflicting information exists, prefer more recent/detailed sources
- Log extraction confidence levels for quality control


### 7. Data Statistics

**GSE**:
- Experiments: 2
- Records: 10
- Avg records/experiment: 5.0

