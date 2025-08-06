import json
import re
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import logging
import pandas as pd
from psycopg2.extensions import connection

logger = logging.getLogger(__name__)


class EnhancedMetadataExtractor:
    """
    Enhanced hierarchical metadata extractor optimized for AI agent processing 
    Handles Study -> Experiment -> Run hierarchy with local PostgreSQl database
    
    Args:
        conn: Database connection object
    """
    
    def __init__(self):
        # Define field mappings for different levels
        self.experiment_level_fields = {
            # Study/Project level info (shared across all runs in an experiment)
            'study_alias', 'study_title', 'study_abstract', 'study_description',
            'gse_title', 'gse_submission_date', 'summary', 'overall_design',
            'pubmed_id', 'organism_ch1', 'scientific_name', 'common_name',
            'library_strategy', 'library_source', 'library_selection',
            'platform', 'instrument_model', 'technology',
            # Publication info
            'study_url_link', 'gse_web_link'
        }

        self.run_level_fields = {
            # Individual run/sample specific info
            'sra_ID', 'run_alias', 'gsm_title', 'characteristics_ch1', 
            'source_name_ch1', 'gsm_submission_date', 'sc_conf_score',
            'spots', 'bases', 'spot_length', 'run_date',
            # Sample specific attributes
            'treatment_protocol_ch1', 'extract_protocol_ch1', 'molecule_ch1'
        }
        
        # Required metadata fields for AI processing
        self.required_extraction_fields = {
            'publication_info': ['pubmed_id', 'study_url_link', 'gse_web_link'],
            'sample_count': ['spots', 'bases', 'gsm_data_row_count'],
            'geographic_info': ['submission_center', 'lab_name', 'center_project_name'],
            'age_info': ['characteristics_ch1', 'treatment_protocol_ch1'],
            'tumor_status': ['characteristics_ch1', 'source_name_ch1', 'gsm_description'],
            'sequencing_method': ['library_strategy', 'library_source', 'platform', 'instrument_model'],
            'tissue_source': ['characteristics_ch1', 'source_name_ch1', 'organism_ch1'],
            'data_access': ['study_xref_link', 'gse_supplementary_file']
        }

    def extract_hierarchical_metadata_from_db(
        self, 
        conn: connection, 
        categorized_data: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Enhanced extraction with database lookup for additional metadata
        
        Args:
            conn: Database connection
            categorized_data: Output from categorize_datasets_by_project
            
        Returns:
            AI-optimized hierarchical metadata structure
        """
        if not categorized_data:
            logger.warning("No categorized data provided")
            return self._create_empty_result()
        
        enhanced_data = {}
        
        for category, records in categorized_data.items():
            if not records or category == 'discarded':
                continue
                
            logger.info(f"Processing {len(records)} records in category: {category}")
            try:
                enhanced_data[category] = self._process_category_with_db_lookup(conn, records)
            except Exception as e:
                logger.error(f"Failed to process category {category}: {e}")
                enhanced_data[category] = self._create_empty_category_result()
        
        return {
            "ai_processing_metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "optimization_strategy": "hierarchical_with_db_enhancement",
                "categories_processed": list(enhanced_data.keys()),
                "total_experiments": sum(data.get('category_summary', {}).get('total_experiments', 0) for data in enhanced_data.values()),
                "extraction_fields": self.required_extraction_fields
            },
            "hierarchical_data": enhanced_data
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        """
        Create empty result structure
        
        Returns:
            Empty result dictionary with metadata
        """
        return {
            "ai_processing_metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "optimization_strategy": "hierarchical_with_db_enhancement",
                "categories_processed": [],
                "total_experiments": 0,
                "extraction_fields": self.required_extraction_fields,
                "status": "no_data"
            },
            "hierarchical_data": {}
        }

    def _create_empty_category_result(self) -> Dict[str, Any]:
        """Create empty category result"""
        return {
            'category_summary': {
                'total_experiments': 0,
                'total_records': 0,
                'avg_records_per_experiment': 0,
                'enhancement_coverage': 0
            },
            'experiments': {}
        }

    def _process_category_with_db_lookup(
        self, 
        conn: connection, 
        records: List[Dict]
    ) -> Dict[str, Any]:
        """
        Enhanced category processing with database lookups
        
        Args:
            conn: Database connection
            records: List of records to process
            
        Returns:
            Processed category data with enhanced metadata
        """
        
        # Group records by experiment
        experiments = defaultdict(list)
        for record in records:
            exp_id = self._get_experiment_id(record)
            experiments[exp_id].append(record)
        
        processed_experiments = {}
        all_sra_ids = [record.get('sra_ID') for record in records if record.get('sra_ID')]
        
        # Get additional metadata from database with error handling
        try:
            enhanced_metadata = self._fetch_enhanced_metadata(conn, all_sra_ids)
        except Exception as e:
            logger.error(f"Failed to fetch enhanced metadata: {e}")
            enhanced_metadata = {}
        
        for exp_id, exp_records in experiments.items():
            try:
                processed_experiments[exp_id] = self._extract_enhanced_experiment_metadata(
                    exp_records, enhanced_metadata
                )
            except Exception as e:
                logger.error(f"Failed to process experiment {exp_id}: {e}")
                processed_experiments[exp_id] = self._create_empty_experiment_result()
        
        return {
            'category_summary': {
                'total_experiments': len(experiments),
                'total_records': len(records),
                'avg_records_per_experiment': len(records) / len(experiments) if experiments else 0,
                'enhancement_coverage': len([r for r in records if r.get('sra_ID') in enhanced_metadata])
            },
            'experiments': processed_experiments
        }

    def _create_empty_experiment_result(self) -> Dict[str, Any]:
        """Create empty experiment result"""
        return {
            'experiment_summary': {
                'total_records': 0,
                'record_types': {},
                'submission_date_range': {'earliest': None, 'latest': None, 'unique_dates': 0},
                'data_volume': {'total_spots': 0, 'total_bases': 0, 'avg_spots_per_record': 0, 'records_with_volume_data': 0}
            },
            'shared_metadata': {},
            'ai_targeted_metadata': {},
            'individual_records': [],
            'ai_processing_instructions': {
                'processing_order': [
                    '1. Process shared_metadata once for experiment context',
                    '2. Use ai_targeted_metadata for structured extraction',
                    '3. Process individual_records for run-specific data'
                ],
                'extraction_priorities': {
                    'high': ['tumor_status', 'sequencing_method', 'tissue_source', 'data_access'],
                    'medium': ['publication_info', 'sample_count', 'geographic_info'],
                    'low': ['age_info']
                }
            }
        }

    def _fetch_enhanced_metadata(
        self, 
        conn: connection, 
        sra_ids: List[str]
    ) -> Dict[str, Dict]:
        """
        Fetch additional metadata from database for better AI processing
        
        Args:
            conn: Database connection
            sra_ids: List of SRA IDs to fetch metadata for
            
        Returns:
            Dictionary mapping sra_ID to enhanced metadata
        """
        if not sra_ids:
            return {}
        
        try:
            # Clean sra_ids (remove .0 suffix if present and handle various formats)
            clean_ids = []
            for sid in sra_ids:
                if sid is None:
                    continue
                sid_str = str(sid)
                if sid_str.replace('.', '').replace('-', '').isdigit():
                    # Handle numeric IDs with potential decimal points
                    try:
                        clean_ids.append(str(int(float(sid_str))))
                    except (ValueError, TypeError):
                        clean_ids.append(sid_str)
                else:
                    clean_ids.append(sid_str)
            
            if not clean_ids:
                return {}
            
            # Remove duplicates
            clean_ids = list(set(clean_ids))
            
            # SQL query for additional metadata with error handling
            placeholders = ','.join(['%s'] * len(clean_ids))
            
            # Use a more conservative query that handles missing columns gracefully
            query = f"""
            SELECT 
                "sra_ID",
                COALESCE(pubmed_id, '') as pubmed_id,
                COALESCE(study_url_link, '') as study_url_link,
                COALESCE(gse_web_link, '') as gse_web_link,
                COALESCE(submission_center, '') as submission_center,
                COALESCE(lab_name, '') as lab_name,
                COALESCE(gsm_description, '') as gsm_description
            FROM merged.sra_geo_ft 
            WHERE CAST("sra_ID" AS TEXT) IN ({placeholders})
            LIMIT 1000
            """
            
            cursor = conn.cursor()
            cursor.execute(query, clean_ids)
            
            enhanced_data = {}
            for row in cursor.fetchall():
                column_names = [desc[0] for desc in cursor.description]
                row_dict = dict(zip(column_names, row))
                sra_id = str(row_dict['sra_id'])
                enhanced_data[sra_id] = row_dict
            
            cursor.close()
            logger.info(f"Fetched enhanced metadata for {len(enhanced_data)} records")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to fetch enhanced metadata: {e}")
            return {}

    def _extract_enhanced_experiment_metadata(
        self, 
        records: List[Dict], 
        enhanced_metadata: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Extract experiment metadata with AI-optimized structure
        
        Args:
            records: All records belonging to same experiment
            enhanced_metadata: Additional metadata from database
            
        Returns:
            AI-optimized experiment metadata structure
        """
        if not records:
            return self._create_empty_experiment_result()
        
        try:
            # Extract shared metadata
            shared_metadata = self._extract_shared_metadata(records)
            
            # Extract AI-targeted metadata
            ai_metadata = self._extract_ai_targeted_metadata(records, enhanced_metadata)
            
            # Process individual records
            individual_records = []
            for i, record in enumerate(records):
                try:
                    record_metadata = self._extract_individual_metadata(record, shared_metadata)
                    
                    # Add enhanced data if available
                    sra_id = str(record.get('sra_ID', ''))
                    if sra_id in enhanced_metadata:
                        record_metadata['enhanced_data'] = enhanced_metadata[sra_id]
                    
                    record_metadata['record_index'] = i
                    individual_records.append(record_metadata)
                except Exception as e:
                    logger.error(f"Failed to process individual record {i}: {e}")
                    continue
            
            return {
                'experiment_summary': {
                    'total_records': len(records),
                    'record_types': self._analyze_record_types(records),
                    'submission_date_range': self._get_date_range(records),
                    'data_volume': self._calculate_data_volume(records)
                },
                'shared_metadata': shared_metadata,
                'ai_targeted_metadata': ai_metadata,
                'individual_records': individual_records,
                'ai_processing_instructions': {
                    'processing_order': [
                        '1. Process shared_metadata once for experiment context',
                        '2. Use ai_targeted_metadata for structured extraction',
                        '3. Process individual_records for run-specific data'
                    ],
                    'extraction_priorities': {
                        'high': ['tumor_status', 'sequencing_method', 'tissue_source', 'data_access'],
                        'medium': ['publication_info', 'sample_count', 'geographic_info'],
                        'low': ['age_info']
                    }
                }
            }
        except Exception as e:
            logger.error(f"Failed to extract enhanced experiment metadata: {e}")
            return self._create_empty_experiment_result()

    def _extract_ai_targeted_metadata(
        self, 
        records: List[Dict], 
        enhanced_metadata: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Extract metadata specifically structured for AI processing
        
        Returns:
            Structured metadata for each required field category
        """
        ai_metadata = {}
        
        try:
            # Combine all available data sources
            all_data = []
            for record in records:
                combined_record = dict(record)
                sra_id = str(record.get('sra_ID', ''))
                if sra_id in enhanced_metadata:
                    combined_record.update(enhanced_metadata[sra_id])
                all_data.append(combined_record)
            
            # Extract each required field category
            for category, field_names in self.required_extraction_fields.items():
                try:
                    ai_metadata[category] = self._extract_category_metadata(all_data, field_names, category)
                except Exception as e:
                    logger.error(f"Failed to extract category {category}: {e}")
                    ai_metadata[category] = {
                        'extraction_fields': field_names,
                        'raw_data': {},
                        'processing_hints': [],
                        'error': str(e)
                    }
        except Exception as e:
            logger.error(f"Failed to extract AI targeted metadata: {e}")
        
        return ai_metadata

    def _extract_category_metadata(
        self, 
        all_data: List[Dict], 
        field_names: List[str], 
        category: str
    ) -> Dict[str, Any]:
        """Extract metadata for a specific category with AI-friendly structure"""
        
        category_data = {
            'extraction_fields': field_names,
            'raw_data': {},
            'processing_hints': []
        }
        
        # Collect raw data from all relevant fields
        for field_name in field_names:
            field_values = []
            for record in all_data:
                value = record.get(field_name)
                if value and str(value).strip() and str(value) not in ['nan', 'None', '']:
                    field_values.append(str(value))
            
            if field_values:
                category_data['raw_data'][field_name] = list(set(field_values))  # Remove duplicates

        # Add category-specific processing hints
        if category == 'tumor_status':
            category_data['processing_hints'] = [
                'Look for keywords: tumor, cancer, normal, healthy, control, malignant',
                'Check tissue descriptions and sample characteristics',
                'Binary classification: tumor=True/False'
            ]
        elif category == 'sequencing_method':
            category_data['processing_hints'] = [
                'Extract: library_strategy, platform, instrument_model',
                'Common values: RNA-Seq, scRNA-seq, Illumina, 10X Genomics'
            ]
        elif category == 'tissue_source':
            category_data['processing_hints'] = [
                'Extract anatomical location and tissue type',
                'Consider organism_ch1 and characteristics_ch1',
                'Map to standard ontology terms if possible'
            ]
        elif category == 'data_access':
            category_data['processing_hints'] = [
                'Look for FTP/HTTP URLs to raw data',
                'Check supplementary file links',
                'Extract accession numbers for data retrieval'
            ]
        
        return category_data

    def _get_experiment_id(self, record: Dict) -> str:
        """Extract experiment identifier with fallback logic"""
        for field in ['study_alias', 'gse_title', 'experiment_alias']:
            if field in record and record[field]:
                value = str(record[field]).strip()
                if value and value not in ['nan', 'None', '']:
                    return value
        
        # Fallback to sra_ID
        return f"exp_{record.get('sra_ID', 'unknown')}"

    def _extract_shared_metadata(self, records: List[Dict]) -> Dict[str, Any]:
        """Extract metadata shared across all records in an experiment"""
        if not records:
            return {}
        
        shared = {}
        first_record = records[0]
        
        for field in self.experiment_level_fields:
            if field in first_record:
                value = first_record[field]
                # Check if all records have the same value for this field
                if all(record.get(field) == value for record in records):
                    shared[field] = value
        
        return shared

    def _extract_individual_metadata(self, record: Dict, shared_metadata: Dict) -> Dict[str, Any]:
        """Extract unique metadata for individual record"""
        individual = {}
        
        for field, value in record.items():
            if field not in shared_metadata:
                individual[field] = value
        
        return individual

    def _analyze_record_types(self, records: List[Dict]) -> Dict[str, int]:
        """Analyze record types for experiment summary"""
        types = defaultdict(int)
        
        for record in records:
            record_type = self._classify_record_type(record)
            types[record_type] += 1
        
        return dict(types)

    def _classify_record_type(self, record: Dict) -> str:
        """Classify individual record type based on characteristics"""
        # Check characteristics and source fields
        check_fields = ['characteristics_ch1', 'source_name_ch1', 'gsm_description']
        
        for field in check_fields:
            if field in record and record[field]:
                text = str(record[field]).lower()
                if any(keyword in text for keyword in ['tumor', 'cancer', 'malignant', 'carcinoma']):
                    return 'tumor'
                elif any(keyword in text for keyword in ['normal', 'healthy', 'control']):
                    return 'normal'
        
        return 'unknown'

    def _get_date_range(self, records: List[Dict]) -> Dict[str, Any]:
        """Get submission date range for records"""
        dates = []
        for record in records:
            for date_field in ['gsm_submission_date', 'submission_date']:
                if date_field in record and record[date_field]:
                    date_str = str(record[date_field])
                    if date_str and date_str not in ['nan', 'None', '']:
                        dates.append(date_str)
        
        if dates:
            return {
                'earliest': min(dates),
                'latest': max(dates),
                'unique_dates': len(set(dates))
            }
        return {'earliest': None, 'latest': None, 'unique_dates': 0}

    def _calculate_data_volume(self, records: List[Dict]) -> Dict[str, Any]:
        """Calculate data volume statistics"""
        total_spots = 0
        total_bases = 0
        valid_records = 0
        
        for record in records:
            if record.get('spots'):
                try:
                    spots_value = float(record['spots'])
                    total_spots += spots_value
                    valid_records += 1
                except (ValueError, TypeError):
                    pass
            
            if record.get('bases'):
                try:
                    bases_value = float(record['bases'])
                    total_bases += bases_value
                except (ValueError, TypeError):
                    pass
        
        return {
            'total_spots': total_spots,
            'total_bases': total_bases,
            'avg_spots_per_record': total_spots / valid_records if valid_records > 0 else 0,
            'records_with_volume_data': valid_records
        }

# Integration functions
def enhance_existing_categorize_workflow(
    conn: connection,
    categorized_data: Dict[str, List[Dict]]
) -> Dict[str, Any]:
    """
    Drop-in enhancement for existing categorize workflow

    Args:
        conn: Database connection object
        categorized_data: Categorized dataset structure from existing workflow
        
    Returns:
        Dictionary containing enhanced hierarchical metadata
    """
    try:
        extractor = EnhancedMetadataExtractor()
        return extractor.extract_hierarchical_metadata_from_db(conn, categorized_data)
    except Exception as e:
        logger.error(f"Failed to enhance categorize workflow: {e}")
        return {
            "ai_processing_metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "optimization_strategy": "hierarchical_with_db_enhancement",
                "categories_processed": [],
                "total_experiments": 0,
                "extraction_fields": {},
                "status": "error",
                "error": str(e)
            },
            "hierarchical_data": {}
        }

def create_ai_ready_export_enhanced(
    conn: connection,
    categorized_data: Dict[str, List[Dict]],
    output_path: str = "ai_enhanced_export.json"
) -> Dict[str, Any]:
    """
    Create enhanced AI-ready export with database integration

    Args:
        conn: Database connection object
        categorized_data: Categorized dataset structure from existing workflow
        output_path: Path to save the enhanced export
        
    Returns:
        Dictionary containing export status and metadata
    """
    try:
        extractor = EnhancedMetadataExtractor()
        hierarchical_data = extractor.extract_hierarchical_metadata_from_db(conn, categorized_data)
        
        # Add processing statistics
        total_experiments = sum(
            data.get('category_summary', {}).get('total_experiments', 0) 
            for data in hierarchical_data.get('hierarchical_data', {}).values()
        )
        
        total_records = sum(
            data.get('category_summary', {}).get('total_records', 0) 
            for data in hierarchical_data.get('hierarchical_data', {}).values()
        )
        
        # Save enhanced export
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchical_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Created enhanced AI export: {output_path}")
        return {
            "status": "success",
            "output_path": output_path,
            "total_experiments": total_experiments,
            "total_records": total_records,
            "enhancement_level": "database_integrated"
        }
        
    except Exception as e:
        logger.error(f"Failed to create enhanced AI export: {e}")
        return {"status": "error", "message": str(e)}


# Local PostgreSQL adapter for existing tools
class LocalPostgreSQLAdapter:
    """
    Adapter to use existing NCBI tools with local PostgreSQL database
    """
    
    def __init__(self, conn: connection):
        self.conn = conn
    
    def get_local_metadata(self, identifiers: List[str], identifier_type: str = 'sra_ID') -> List[Dict]:
        """
        Fetch metadata from local database using various identifiers
        
        Args:
            identifiers: List of identifiers (sra_ID, study_alias, etc.)
            identifier_type: Type of identifier being used
            
        Returns:
            List of metadata dictionaries
        """
        try:
            placeholders = ','.join(['%s'] * len(identifiers))
            
            # Map identifier types to database columns
            column_mapping = {
                'sra_ID': 'sra_ID',
                'study_alias': 'study_alias', 
                'experiment_alias': 'experiment_alias',
                'run_alias': 'run_alias',
                'gsm_title': 'gsm_title'
            }
            
            column = column_mapping.get(identifier_type, 'sra_ID')
            
            query = f"""
            SELECT *
            FROM merged.sra_geo_ft 
            WHERE {column} IN ({placeholders})
            ORDER BY gsm_submission_date DESC
            """
            
            cursor = self.conn.cursor()
            cursor.execute(query, identifiers)
            
            results = []
            column_names = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                results.append(dict(zip(column_names, row)))
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch local metadata: {e}")
            return []

    def get_related_records(self, base_identifiers: List[str]) -> Dict[str, List[Dict]]:
        """
        Get related records (same study/experiment) for given identifiers
        
        Returns:
            Dictionary with relationship mappings
        """
        try:
            # Get study_alias for base identifiers
            placeholders = ','.join(['%s'] * len(base_identifiers))
            
            query = f"""
            SELECT DISTINCT study_alias, sra_ID
            FROM merged.sra_geo_ft 
            WHERE sra_ID IN ({placeholders})
            """
            
            cursor = self.conn.cursor()
            cursor.execute(query, base_identifiers)
            
            study_mappings = {}
            for row in cursor.fetchall():
                study_alias, sra_id = row
                if study_alias not in study_mappings:
                    study_mappings[study_alias] = []
                study_mappings[study_alias].append(str(sra_id))
            
            # Get all records for these studies
            all_studies = list(study_mappings.keys())
            if not all_studies:
                return {}
            
            study_placeholders = ','.join(['%s'] * len(all_studies))
            query = f"""
            SELECT *
            FROM merged.sra_geo_ft 
            WHERE study_alias IN ({study_placeholders})
            ORDER BY study_alias, gsm_submission_date
            """
            
            cursor.execute(query, all_studies)
            column_names = [desc[0] for desc in cursor.description]
            
            related_records = defaultdict(list)
            for row in cursor.fetchall():
                record = dict(zip(column_names, row))
                study = record['study_alias']
                related_records[study].append(record)
            
            cursor.close()
            return dict(related_records)
            
        except Exception as e:
            logger.error(f"Failed to get related records: {e}")
            return {}


# Testing function
if __name__ == "__main__":
    # Simple test
    print("=== Testing Enhanced Metadata Extractor ===")
    
    # Create sample categorized data
    sample_data = {
        'GSE': [
            {'study_alias': 'GSE123456', 'sra_ID': '001', 'gsm_title': 'Sample 1'},
            {'study_alias': 'GSE123456', 'sra_ID': '002', 'gsm_title': 'Sample 2'}
        ],
        'PRJNA': [
            {'study_alias': 'PRJNA789012', 'sra_ID': '003', 'gsm_title': 'Sample 3'}
        ]
    }
    
    extractor = EnhancedMetadataExtractor()
    
    # Test without database connection (should handle gracefully)
    result = extractor.extract_hierarchical_metadata_from_db(None, sample_data)
    
    print(f"✅ Test completed")
    print(f"📊 Categories processed: {result['ai_processing_metadata']['categories_processed']}")
    print(f"🧪 Total experiments: {result['ai_processing_metadata']['total_experiments']}")
