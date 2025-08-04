from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import json
import logging

logger = logging.getLogger(__name__)

@tool
def get_local_study_metadata(
    study_identifiers: Annotated[List[str], "List of study identifiers (study_alias, GSE IDs)"],
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of local study metadata"]:
    """
    Get study-level metadata from local PostgreSQL database.
    Adapted from bigquery.py get_study_metadata for local use.
    """
    if config is None or not config.get("configurable", {}).get("db_conn"):
        return "No database connection provided."
    
    conn = config["configurable"]["db_conn"]
    adapter = LocalPostgreSQLAdapter(conn)
    
    try:
        results = adapter.get_local_metadata(study_identifiers, 'study_alias')
        
        # Group by study and aggregate
        study_data = {}
        for record in results:
            study_id = record.get('study_alias')
            if study_id not in study_data:
                study_data[study_id] = {
                    'study_alias': study_id,
                    'study_title': record.get('study_title'),
                    'summary': record.get('summary'),
                    'experiment_count': 0,
                    'experiments': []
                }
            
            study_data[study_id]['experiment_count'] += 1
            study_data[study_id]['experiments'].append({
                'sra_ID': record.get('sra_ID'),
                'gsm_title': record.get('gsm_title'),
                'organism': record.get('organism_ch1')
            })
        
        return json.dumps(list(study_data.values()), indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Failed to get local study metadata: {e}")
        return f"Error: {str(e)}"

@tool 
def get_local_experiment_metadata(
    experiment_identifiers: Annotated[List[str], "List of experiment identifiers (sra_ID, experiment_alias)"],
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of local experiment metadata"]:
    """
    Get experiment-level metadata from local PostgreSQL database.
    Adapted from bigquery.py get_experiment_metadata for local use.
    """
    if config is None or not config.get("configurable", {}).get("db_conn"):
        return "No database connection provided."
    
    conn = config["configurable"]["db_conn"]
    adapter = LocalPostgreSQLAdapter(conn)
    
    try:
        results = adapter.get_local_metadata(experiment_identifiers, 'sra_ID')
        
        processed_results = []
        for record in results:
            processed_results.append({
                'sra_ID': record.get('sra_ID'),
                'experiment_alias': record.get('experiment_alias'),
                'library_strategy': record.get('library_strategy'),
                'library_source': record.get('library_source'),
                'platform': record.get('platform'),
                'instrument_model': record.get('instrument_model'),
                'organism': record.get('organism_ch1'),
                'tissue_info': record.get('characteristics_ch1'),
                'study_info': {
                    'study_alias': record.get('study_alias'),
                    'study_title': record.get('study_title')
                }
            })
        
        return json.dumps(processed_results, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Failed to get local experiment metadata: {e}")
        return f"Error: {str(e)}"

@tool
def get_local_run_metadata(
    run_identifiers: Annotated[List[str], "List of run identifiers (sra_ID, run_alias)"],
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of local run metadata"]:
    """
    Get run-level metadata from local PostgreSQL database.
    Adapted from bigquery.py get_run_metadata for local use.
    """
    if config is None or not config.get("configurable", {}).get("db_conn"):
        return "No database connection provided."
    
    conn = config["configurable"]["db_conn"]
    adapter = LocalPostgreSQLAdapter(conn)
    
    try:
        results = adapter.get_local_metadata(run_identifiers, 'sra_ID')
        
        processed_results = []
        for record in results:
            # Calculate data volume
            spots = record.get('spots', 0)
            bases = record.get('bases', 0)
            spot_length = record.get('spot_length', 0)
            
            try:
                spots = float(spots) if spots else 0
                bases = float(bases) if bases else 0
                spot_length = int(spot_length) if spot_length else 0
            except (ValueError, TypeError):
                spots = bases = spot_length = 0
            
            processed_results.append({
                'sra_ID': record.get('sra_ID'),
                'run_alias': record.get('run_alias'),
                'experiment_alias': record.get('experiment_alias'),
                'organism': record.get('organism_ch1'),
                'tissue_characteristics': record.get('characteristics_ch1'),
                'data_volume': {
                    'spots': spots,
                    'bases': bases,
                    'spot_length': spot_length,
                    'estimated_reads': spots * 2 if spot_length > 0 else 0  # Assuming paired-end
                },
                'submission_info': {
                    'submission_date': record.get('gsm_submission_date'),
                    'submission_center': record.get('submission_center')
                }
            })
        
        return json.dumps(processed_results, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Failed to get local run metadata: {e}")
        return f"Error: {str(e)}"

@tool
def extract_structured_metadata(
    identifiers: Annotated[List[str], "List of identifiers (any level: study, experiment, run)"],
    extraction_fields: Annotated[List[str], "Fields to extract: publication_info, sample_count, geographic_info, age_info, tumor_status, sequencing_method, tissue_source, data_access"] = None,
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of structured metadata optimized for AI processing"]:
    """
    Extract structured metadata from local database optimized for AI agent processing.
    This tool provides the exact metadata fields needed for downstream analysis.
    """
    if config is None or not config.get("configurable", {}).get("db_conn"):
        return "No database connection provided."
    
    if extraction_fields is None:
        extraction_fields = ["tumor_status", "sequencing_method", "tissue_source", "data_access"]
    
    conn = config["configurable"]["db_conn"]
    adapter = LocalPostgreSQLAdapter(conn)
    
    try:
        # Get all available metadata for identifiers
        results = adapter.get_local_metadata(identifiers, 'sra_ID')
        
        if not results:
            return json.dumps({"error": "No records found for provided identifiers"})
        
        # Initialize extractor
        extractor = EnhancedMetadataExtractor()
        
        structured_output = {
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "identifiers_processed": identifiers,
                "records_found": len(results),
                "extraction_fields": extraction_fields
            },
            "structured_data": {}
        }
        
        # Extract each requested field category
        for field_category in extraction_fields:
            if field_category in extractor.required_extraction_fields:
                field_names = extractor.required_extraction_fields[field_category]
                category_data = extractor._extract_category_metadata(results, field_names, field_category)
                structured_output["structured_data"][field_category] = category_data
        
        # Add summary statistics
        structured_output["summary"] = {
            "unique_studies": len(set(r.get('study_alias') for r in results if r.get('study_alias'))),
            "unique_organisms": len(set(r.get('organism_ch1') for r in results if r.get('organism_ch1'))),
            "date_range": extractor._get_date_range(results),
            "data_volume": extractor._calculate_data_volume(results)
        }
        
        return json.dumps(structured_output, indent=2, default=str, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Failed to extract structured metadata: {e}")
        return json.dumps({"error": str(e)})

@tool
def get_data_access_urls(
    identifiers: Annotated[List[str], "List of SRA IDs or study identifiers"],
    config: RunnableConfig = None,
) -> Annotated[str, "JSON string of data access URLs and download information"]:
    """
    Get data access URLs and download information for raw sequencing data.
    Returns FTP/HTTP links to FASTQ/BAM files.
    """
    if config is None or not config.get("configurable", {}).get("db_conn"):
        return "No database connection provided."
    
    conn = config["configurable"]["db_conn"]
    
    try:
        # Query for data access information
        placeholders = ','.join(['%s'] * len(identifiers))
        query = f"""
        SELECT 
            sra_ID,
            run_alias,
            study_alias,
            experiment_alias,
            -- Data access fields
            study_xref_link,
            gse_supplementary_file,
            experiment_url_link,
            -- Additional info for constructing URLs
            platform,
            spots,
            bases,
            library_layout
        FROM merged.sra_geo_ft 
        WHERE sra_ID IN ({placeholders}) OR study_alias IN ({placeholders})
        """
        
        cursor = conn.cursor()
        cursor.execute(query, identifiers + identifiers)  # Duplicate for both conditions
        
        data_access_info = []
        column_names = [desc[0] for desc in cursor.description]
        
        for row in cursor.fetchall():
            record = dict(zip(column_names, row))
            
            access_info = {
                "sra_ID": record.get('sra_id'),
                "run_alias": record.get('run_alias'),
                "study_alias": record.get('study_alias'),
                "experiment_alias": record.get('experiment_alias'),
                "data_access": {
                    "study_link": record.get('study_xref_link'),
                    "supplementary_files": record.get('gse_supplementary_file'),
                    "experiment_link": record.get('experiment_url_link')
                },
                "technical_info": {
                    "platform": record.get('platform'),
                    "library_layout": record.get('library_layout'),
                    "estimated_file_size_gb": _estimate_file_size(record.get('bases'))
                },
                "suggested_download_urls": _generate_download_urls(record)
            }
            
            data_access_info.append(access_info)
        
        cursor.close()
        
        result = {
            "data_access_summary": {
                "total_records": len(data_access_info),
                "records_with_links": len([r for r in data_access_info if any(r["data_access"].values())]),
                "estimated_total_size_gb": sum(_estimate_file_size(r.get("technical_info", {}).get("bases", 0)) for r in data_access_info)
            },
            "access_information": data_access_info
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Failed to get data access URLs: {e}")
        return json.dumps({"error": str(e)})

def _estimate_file_size(bases):
    """Estimate FASTQ file size based on number of bases"""
    if not bases:
        return 0
    try:
        bases_float = float(bases)
        # Rough estimation: ~1 byte per base for compressed FASTQ
        size_gb = bases_float / (1024**3)
        return round(size_gb, 2)
    except (ValueError, TypeError):
        return 0

def _generate_download_urls(record):
    """Generate suggested download URLs based on available information"""
    urls = []
    
    sra_id = record.get('sra_id')
    run_alias = record.get('run_alias')
    
    if sra_id:
        # Standard SRA download URLs
        sra_id_clean = str(sra_id).replace('.0', '')
        
        # NCBI SRA FTP URL pattern
        if sra_id_clean.startswith('SRR'):
            # SRA FTP URL construction
            srr_prefix = sra_id_clean[:6]  # e.g., SRR123
            urls.append({
                "type": "SRA_FTP",
                "url": f"ftp://ftp.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByRun/sra/{sra_id_clean[:3]}/{srr_prefix}/{sra_id_clean}/{sra_id_clean}.sra",
                "description": "Direct SRA file download"
            })
            
            # European Nucleotide Archive (ENA) alternative
            urls.append({
                "type": "ENA_FTP", 
                "url": f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{sra_id_clean[:6]}/{sra_id_clean}/{sra_id_clean}.fastq.gz",
                "description": "ENA FASTQ download (if available)"
            })
    
    # Add supplementary file URLs if available
    supp_files = record.get('gse_supplementary_file')
    if supp_files:
        urls.append({
            "type": "SUPPLEMENTARY",
            "url": supp_files,
            "description": "GEO supplementary files"
        })
    
    return urls

# Import required classes at the top
from datetime import datetime