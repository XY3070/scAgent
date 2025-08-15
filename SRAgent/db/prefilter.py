import warnings
import pandas as pd
import warnings
import logging
import re
from typing import List, Dict, Any, Optional, Set
from psycopg2.extensions import connection
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import json


logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """
    Filtered results data class.
    """
    data: pd.DataFrame
    count: int
    filter_name: str
    description: str
    reduction_count: int = 0
    reduction_pct: float = 0.0
    
    def __post_init__(self):
        """
        Calculate the number of records.
        """
        if self.data is not None:
            self.count = len(self.data)
    
    def log_result(self, previous_count: int = None):
        """
        Log the filter results.
        """
        if previous_count is not None:
            self.reduction_count = previous_count - self.count
            self.reduction_pct = (self.reduction_count / previous_count * 100) if previous_count > 0 else 0
            
        # Main result line
        print(f"📊 {self.filter_name}: {self.count:,} records "
            f"(↓{self.reduction_count:,}, -{self.reduction_pct:.1f}%)")
        if self.description:
            print(f"   {self.description}")
        print()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get the summary of the filter results.
        """
        return {
            "filter_name": self.filter_name,
            "count": self.count,
            "reduction_count": self.reduction_count,
            "reduction_pct": self.reduction_pct
        }


class BaseFilter(ABC):
    """
    Base filter class.
    """
    
    def __init__(self, conn: connection, name: str = None):
        self.conn = conn
        self.name = name or self.__class__.__name__
        
    @abstractmethod
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Apply the filter.
        """
        pass

    def execute_query_with_cursor(self, query: str, params: tuple) -> pd.DataFrame:
        """
        Execute the query and return a DataFrame.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            if cursor.description is None:
                cursor.close()
                return pd.DataFrame()
            
            colnames = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            
            return pd.DataFrame(results, columns=colnames)
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            try:
                cursor.close()
            except:
                pass
            return pd.DataFrame()


class InitialDatasetFilter(BaseFilter):
    """
    Initial dataset filter class.
    """
    
    def __init__(self, conn: connection, table_name: str = "merged.sra_geo_ft"):
        super().__init__(conn, "Initial Dataset")
        self.table_name = table_name
    
    def apply(self, input_result: FilterResult = None) -> FilterResult:
        """
        Get the initial dataset.
        """

        def _execute():
            # First, check what columns actually exist
            check_query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = %s
                ORDER BY column_name
            """

            schema, table = self.table_name.split('.') if '.' in self.table_name else ('public', self.table_name)
            avail_col_df = self.execute_query_with_cursor(check_query, (table, schema))

            if avail_col_df.empty:
                return FilterResult(
                    data=pd.DataFrame(),
                    count=0,
                    filter_name=self.name,
                    description="Table not found or inaccessible",
                )

            avail_col = set(avail_col_df['column_name'].tolist())

            # Define desired cols and check availability 
            desired_cols = [
                # coral identifier
                'sra_ID', 'study_alias', 'experiment_name', 'run_alias', 'sample_alias', 
                # title and description
                'study_title', 'study_abstract', 'study_description', 'experiment_title', 
                'gse_title', 'gsm_title', 'summary', 'overall_design', 'design_description', 
                # biological info
                'scientific_name', 'organism_ch1', 'common_name', 'source_name_ch1',
                'characteristics_ch1', 'description',
                # tech info
                'library_strategy', 'library_source', 'library_selection', 'library_layout', 
                'technology', 'platform', 'instrument_model', 'platoform_parameters', 
                # time info  
                'gsm_submission_date', 'submission_date', 'run_date', 
                # single cell info
                'sc_conf_score', 
                # attribute info  
                'sample_attribute', 'experiment_attribute', 'study_attribute', 'run_attribute', 
                # other info
                'pubmed_id', 'center_project_name', 'spots', 'bases'
            ]

            # Select only columns that exist
            existing_cols = [col for col in desired_cols if col in avail_col]
            missing_cols = [col for col in desired_cols if col not in avail_col]
            
            # Build query with existing columns
            columns_str = ', '.join([f'"{col}"' for col in existing_cols])
            query = f"SELECT {columns_str} FROM {self.table_name}"
            
            df = self.execute_query_with_cursor(query, ())

            return FilterResult(
                data=df,
                count=len(df),
                filter_name=self.name,
                description="All records in database",
            )

        result = _execute()
        result.log_result()
        return result


class BasicAvailabilityFilter(BaseFilter):
    """
    Basic availability filter class.
    """
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter records with basic data.
        """
        if input_result.data.empty:
            return FilterResult(
                data=pd.DataFrame(),
                count=0,
                filter_name="Basic Availability",
                description="No input data"
            )
        
        # Filter records with basic data in memory, instead of re-querying the database
        # create separate masks for clarity; check if cols exist first
        sra_id_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        if 'sra_ID' in input_result.data.columns:  
            sra_id_mask = (
                input_result.data['sra_ID'].notna() & 
                input_result.data['sra_ID'] != ''
            )
        
        study_alias_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        if 'study_alias' in input_result.data.columns:
            study_alias_mask = (
                input_result.data['study_alias'].notna() & 
                input_result.data['study_alias'] != ''
            )

        gse_title_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        if 'gse_title' in input_result.data.columns:
            gse_title_mask = (
                input_result.data['gse_title'].notna() & 
                input_result.data['gse_title'] != ''
            )
        
        combined_mask = sra_id_mask | study_alias_mask | gse_title_mask
        filtered_df = input_result.data[combined_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Basic Availability",
            description="Has SRA_ID or GSE_ID"
        )
        
        result.log_result(input_result.count)
        return result


class OrganismFilter(BaseFilter):
    """
    Organism filter class.
    """
    
    # Common false positive keywords that should be excluded even if they match human
    EXCLUDE_PATTERNS = [
        r'\bmouse\b', r'\bmurine\b', r'\bmus musculus\b',  # Mouse-related terms
        r'\brat\b', r'\brattus\b', r'\brattus norvegicus\b',  # Rat-related terms
        r'\bfruit fly\b', r'\bdrosophila\b', r'\bdrosophila melanogaster\b',  # Fruit fly
        r'\bzebrafish\b', r'\bdanio rerio\b',  # Zebrafish
        r'\bmonkey\b', r'\bmacaque\b', r'\brhesus\b',  # Non-human primates
        r'\bmodel organism\b',  # Generic model organism mentions
        r'\bcell line\b',  # Cell lines that might be human but are not primary tissue
        r'\bhepg2\b', r'\bhek293\b', r'\bhela\b',  # Common human cell lines
        r'\bxenograft\b',  # Xenograft models (human cells in other organisms)
        r'\bhumanized\b',  # Humanized models (modified organisms)
        r'\bsars-cov\b', r'\bsars cov\b', r'\bcovid\b', r'\bsars-2\b', r'\bnovel coronavirus\b',  # Viruses often associated with humans
        r'\bh1n1\b', r'\bh3n2\b', r'\binfluenza\b',  # Influenza viruses
        r'\bhiv\b', r'\bhepatitis\b', r'\bebv\b', r'\bepstein-barr\b',  # Other human-associated viruses
        r'\bvirus\b.*\bhomo sapiens\b', r'\bhomo sapiens\b.*\bvirus\b'  # Virus and human co-occurrence patterns
    ]
    
    def __init__(self, conn: connection, organisms: List[str]):
        super().__init__(conn)
        self.organisms = organisms
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter records with specified organisms.
        """
        if input_result.data.empty or not self.organisms:
            return input_result
            
        if "human" not in [org.lower() for org in self.organisms]:
            return input_result
        
        # Filter records with specified organisms in memory
        # Check if each column exists before using it
        human_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        
        if 'organism_ch1' in input_result.data.columns:
            human_mask |= input_result.data['organism_ch1'].str.contains('homo sapiens', case=False, na=False)
            
        if 'scientific_name' in input_result.data.columns:
            human_mask |= input_result.data['scientific_name'].str.contains('homo sapiens', case=False, na=False)
            
        if 'organism' in input_result.data.columns:
            human_mask |= input_result.data['organism'].str.contains('homo sapiens', case=False, na=False)
            
        if 'source_name_ch1' in input_result.data.columns:
            human_mask |= input_result.data['source_name_ch1'].str.contains('human', case=False, na=False)
            
        if 'common_name' in input_result.data.columns:
            human_mask |= input_result.data['common_name'].str.contains('human', case=False, na=False)
        
        # Create exclusion mask for false positive keywords (only in the same 5 columns used for human matching)
        exclude_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        human_matching_columns = ['organism_ch1', 'scientific_name', 'organism', 'source_name_ch1', 'common_name']
        
        for pattern in self.EXCLUDE_PATTERNS:
            for col in human_matching_columns:
                if col in input_result.data.columns:
                    exclude_mask |= input_result.data[col].str.contains(pattern, case=False, na=False, regex=True)
        
        # Apply both inclusion and exclusion filters
        final_mask = human_mask & ~exclude_mask
        filtered_df = input_result.data[final_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Organism Filter",
            description="Human samples only (excluded common false positives including viruses)"
        )
        
        result.log_result(input_result.count)
        return result


class SingleCellFilter(BaseFilter):
    """
    Single cell filter class.
    """
    
    def __init__(self, conn: connection, min_confidence: int = 2, use_fallback: bool = True):
        super().__init__(conn, "Single Cell Filter")
        self.min_confidence = min_confidence
        self.use_fallback = use_fallback
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter single cell data.
        """

        if input_result.data.empty:
            return FilterResult(
                data=pd.DataFrame(),
                count=0,
                filter_name=self.name,
                description="No input data"
            )

        def _execute():
            filtered_df = input_result.data.copy()
            strategy_used = "None"
            warnings = []  # iniialize warnings list
            metadata = {'strategy': 'none'}  # initialize metadata
            
            # Strategy 1: Use sc_conf_score if available
            if 'sc_conf_score' in input_result.data.columns:
                valid_scores = input_result.data['sc_conf_score'].notna()
                if valid_scores.any():
                    score_stats = input_result.data.loc[valid_scores, 'sc_conf_score'].describe()
                    sc_mask = (
                        input_result.data['sc_conf_score'].notna() & 
                        (input_result.data['sc_conf_score'] >= self.min_confidence)
                    )
                    filtered_df = input_result.data[sc_mask].copy()
                    strategy_used = f"sc_conf_score >= {self.min_confidence}"
                    metadata = {'strategy': 'score_based'}
                    
                    if len(filtered_df) > 0:
                        return filtered_df, strategy_used, metadata
                    else:
                        warnings.append(f"No records with sc_conf_score >= {self.min_confidence}")
            
            # Strategy 2: Fallback to text-based detection
            if self.use_fallback and len(filtered_df) == 0:
                sc_tech_patterns = [
                    r'10[xX]', r'\b10x\b', 'Chromium', 'SMART-seq2', 'Drop-seq',
                    'microwell', r'C1\s?System', 'single cell', 'scRNA-seq',
                    'snRNA-seq', 'CITE-seq', 'single-cell'
                ]
                
                include_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
                text_columns = ['overall_design', 'library_strategy', 'summary', 'study_title']
                
                matched_patterns = []
                for pattern in sc_tech_patterns:
                    for col in text_columns:
                        if col in input_result.data.columns:
                            pattern_mask = input_result.data[col].str.contains(
                                pattern, case=False, na=False, regex=True
                            )
                            if pattern_mask.any():
                                matched_patterns.append(f"{pattern} in {col}")
                                include_mask |= pattern_mask
                
                if include_mask.any():
                    filtered_df = input_result.data[include_mask].copy()
                    strategy_used = "Text pattern matching"
                    metadata = {'strategy': 'text_based', 'patterns': matched_patterns}
                else:
                    warnings.append("No single-cell patterns found in text fields")
                    metadata = {'strategy': 'failed'}
            
            return filtered_df, strategy_used, metadata
        
        result_data, strategy, metadata = _execute()
        
        result = FilterResult(
            data=result_data,
            count=len(result_data),
            filter_name=self.name,
            description=strategy
        )
        
        result.log_result(input_result.count)
        return result


class ExclusionSingleCellFilter(BaseFilter):
    """
    Exclusion-based single cell filter class.
    """

    # Keywords that strongly indicate non-single cell data
    # Balanced approach: Keep definite non-single-cell keywords and make high-risk ones more specific
    EXCLUDE_PATTERNS = [
        # Definite bulk technologies (should always be excluded)
        r'\bbulk RNA\b', r'\bbulk sequencing\b', r'\bbulk expression\b',
        # Definite non-RNA-seq technologies (should always be excluded)
        r'\bmicroarray\b', r'\bqPCR\b', r'\bRT-qPCR\b', r'\bquantitative PCR\b',
        # Definite other omics (should always be excluded)
        r'\bChIP-seq\b', r'\bHi-C\b', r'\b[45]C\b', r'\bcapture C\b',
        r'\bwhole exome sequencing\b', r'\bWES\b', r'\bwhole genome sequencing\b', r'\bWGS\b',
        r'\bDNA-seq\b', 
        # ATAC-seq needs special handling because scATAC-seq exists
        r'\bATAC-seq\s+(?=.*(bulk|pool))',  # Only exclude bulk ATAC-seq
        r'\bmetagenomic\b', r'\bamplicon sequencing\b', r'\b16S rRNA\b',
        # Definite bulk sample preparations (should always be excluded)
        r'\bpooled sequencing\b', r'\bpopulation sequencing\b', r'\bmixed culture\b',
        r'\btissue homogenate\b', r'\bcell lysate\b', r'\bwhole tissue\b',
        # Definite other technologies (should always be excluded)
        r'\bflow cytometry\b', r'\bFACS\b', r'\bmass spectrometry\b', r'\bproteomics\b',
        r'\bimmunohistochemistry\b', r'\bwestern blot\b', r'\bELISA\b',
        r'\bmicroscopy\b', r'\bimaging\b', r'\bhistology\b',
        # Definite study contexts that are not single-cell
        r'\bpatient derived xenograft\b', r'\bPDX\b',
        r'\bclinical trial\b', r'\bcohort study\b', r'\bepidemiological study\b',
        r'\breview article\b', r'\bmeta-analysis\b', r'\bdatabase\b',
        r'\bsimulation\b', r'\bcomputational model\b', r'\balgorithm\b',
        r'\bsynthetic data\b', r'\bspike-in\b', r'\bcontrol sample\b',
        # More specific patterns for high-risk terms to reduce false negatives
        # Keep these but make them more specific to avoid excluding legitimate single-cell data
        r'\b(cell line|cell lines)\s+(?=.*(bulk|microarray|qPCR|ChIP|proteom))',  # Only exclude cell lines in bulk contexts
        r'\borganoid\s+(?=.*(bulk|microarray|qPCR|ChIP|proteom))',  # Only exclude organoids in bulk contexts
        r'\bspheroid\s+(?=.*(bulk|microarray|qPCR|ChIP|proteom))',  # Only exclude spheroids in bulk contexts
        r'\bcell culture\s+(?=.*(bulk|microarray|qPCR|ChIP|proteom))',  # Only exclude cell culture in bulk contexts
        # Animal model is kept in a separate filter context, not here
    ]

    def __init__(self, conn: connection):
        super().__init__(conn, "Exclusion Single Cell Filter")

    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter records by excluding non-single cell keywords.
        """
        if input_result.data.empty:
            return input_result

        # Define all relevant text columns from the provided list
        text_columns = [
            'study_title', 'summary', 'overall_design', 'scientific_name',
            'library_strategy', 'technology', 'characteristics_ch1', 'gse_title',
            'gsm_title', 'organism_ch1', 'source_name_ch1', 'common_name',
            'experiment_name', 'experiment_title', 'design_description',
            'library_name', 'library_selection', 'library_construction_protocol',
            'platform', 'instrument_model', 'platform_parameters',
            'experiment_attribute', 'sample_alias', 'description',
            'sample_attribute', 'study_abstract', 'center_project_name',
            'study_description', 'study_attribute', 'submission_comment',
            'gsm_description', 'data_processing', 'overall_design'
        ]

        exclude_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)

        for pattern in self.EXCLUDE_PATTERNS:
            for col in text_columns:
                if col in input_result.data.columns:
                    # Use .fillna('') to treat NaN as empty strings for regex matching
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        exclude_mask |= input_result.data[col].fillna('').str.contains(pattern, flags=re.IGNORECASE, na=False, regex=True)

        # Keep records that are NOT in the exclude_mask
        filtered_df = input_result.data[~exclude_mask].copy()

        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name=self.name,
            description="Excluded records based on non-single cell keywords"
        )

        result.log_result(input_result.count)
        return result


class CellLineFilter:
    """
    Cell line filter class.
    """
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter samples that are cell lines.
        """
        if input_result.data.empty:
            return input_result
        
        # Cell line keywords
        cell_line_patterns = [
            r'\b(?:cell line|immortalized|HEK293|HeLa|Jurkat|CHO|NIH/3T3|K562|U2OS|HEPG2|A549)\b'
        ]
        
        # Filter records that are cell lines in memory
        cell_line_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        
        text_columns = ['overall_design', 'characteristics_ch1', 'summary', 'source_name_ch1', 'sample_attribute']
        
        for pattern in cell_line_patterns:
            for col in text_columns:
                if col in input_result.data.columns:
                    cell_line_mask |= input_result.data[col].str.contains(pattern, case=False, na=False, regex=True)
        
        filtered_df = input_result.data[cell_line_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Cell Line Filter",
            description="Samples identified as cell lines"
        )
        
        result.log_result(input_result.count)
        return result


class SequencingStrategyFilter(BaseFilter):                                                                                                                                             
    """
    Sequencing strategy filter class.
    """

    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter single cell sequencing technologies.
        """
        if input_result.data.empty:
            return input_result
            
        # Define single cell sequencing technologies
        sc_tech_patterns = [
            r'10[xX]', r'\b10x\b', 'Chromium', 'SMART-seq2', 'Drop-seq',
            'microwell', r'C1\s?System', 'Single cell sequencing', 'scRNA-seq',
            'snRNA-seq', 'CITE-seq'
        ]
        
        # Exclude keywords
        exclude_patterns = ['bulk', 'microorganism', 'yeast', r'E\. coli', 'bulk RNA', 'tissue profiling']
        
        # Filter records with single cell sequencing technologies in memory
        include_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        exclude_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        
        for pattern in sc_tech_patterns:
            if 'overall_design' in input_result.data.columns:
                include_mask |= input_result.data['overall_design'].str.contains(pattern, case=False, na=False, regex=True)
            if 'library_strategy' in input_result.data.columns:
                include_mask |= input_result.data['library_strategy'].str.contains(pattern, case=False, na=False, regex=True)
            if 'summary' in input_result.data.columns:
                include_mask |= input_result.data['summary'].str.contains(pattern, case=False, na=False, regex=True)
                
        for pattern in exclude_patterns:
            if 'overall_design' in input_result.data.columns:
                exclude_mask |= input_result.data['overall_design'].str.contains(pattern, case=False, na=False, regex=True)
            if 'summary' in input_result.data.columns:
                exclude_mask |= input_result.data['summary'].str.contains(pattern, case=False, na=False, regex=True)
                
        final_mask = include_mask & ~exclude_mask
        filtered_df = input_result.data[final_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Sequencing Strategy",
            description="Single-cell sequencing technologies"
        )
        
        result.log_result(input_result.count)
        return result


class CancerStatusFilter(BaseFilter):
    """
    Cancer status filter class.
    """
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter cancer or normal samples.
        """
        if input_result.data.empty:
            return input_result
        
        # Cancer-related keywords
        cancer_patterns = [
            'tumor', 'cancer', 'carcinoma', 'neoplasm', 'malignant', 'metastatic', 
            'onco', 'glioma', 'sarcoma', 'leukemia', 'lymphoma', 'adenocarcinoma', 
            'melanoma', 'osteosarcoma', 'neuroblastoma'
        ]
        
        # Normal sample keywords
        normal_patterns = [
            'normal', 'non-?tumor', 'healthy', 'control', 'donor', 'peripheral blood', 
            'adjacent normal', 'para-tumor', 'healthy donor'
        ]
        
        # Filter records with cancer or normal samples in memory
        cancer_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        normal_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        
        text_columns = ['overall_design', 'summary', 'study_title', 'characteristics_ch1']
        
        for pattern in cancer_patterns:
            for col in text_columns:
                if col in input_result.data.columns:
                    cancer_mask |= input_result.data[col].str.contains(pattern, case=False, na=False, regex=True)
        
        for pattern in normal_patterns:
            for col in text_columns:
                if col in input_result.data.columns:
                    normal_mask |= input_result.data[col].str.contains(pattern, case=False, na=False, regex=True)
        
        final_mask = cancer_mask | normal_mask
        filtered_df = input_result.data[final_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Cancer Status",
            description="Cancer or normal tissue samples"
        )
        
        result.log_result(input_result.count)
        return result

class TissueSourceFilter(BaseFilter):
    """
    Tissue source filter class.
    """
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter samples from major tissue systems.
        """
        if input_result.data.empty:
            return input_result
        
        # Major tissue system keywords (non-capturing groups)
        tissue_patterns = [
            # Neural
            r'\b(?:brain|cerebral|hippocampus|cortex|striatum|cerebellum|spinal cord|retina|optic nerve|glia|neuron)\b',
            # Respiratory
            r'\b(?:lung|pulmonary|alveolar|bronchial|trachea|nasal|larynx|pharynx|diaphragm)\b',
            # Digestive
            r'\b(?:liver|hepatic|gastric|stomach|intestine|colon|ileum|jejunum|duodenum|esophagus|pancreas|gallbladder|bile duct)\b',
            # Circulatory
            r'\b(?:heart|cardiac|myocardial|vascular|artery|vein|capillary|blood|pbmc|plasma|serum)\b',
            # Immune
            r'\b(?:lymph node|spleen|thymus|bone marrow|tonsil|macrophage|lymphocyte|neutrophil|dendritic cell|mast cell)\b',
            # Urinary
            r'\b(?:kidney|renal|nephron|bladder|ureter|urethra)\b',
            # Reproductive
            r'\b(?:ovary|testis|uterus|placenta|prostate|penis|vagina|fallopian tube|endometrium)\b',
            # Endocrine
            r'\b(?:thyroid|parathyroid|adrenal|pituitary|pancreatic islet|hypothalamus)\b',
            # Skin
            r'\b(?:skin|epidermis|dermis|melanocyte|hair follicle|sweat gland|sebaceous gland)\b',
            # Muscle
            r'\b(?:muscle|skeletal muscle|cardiac muscle|smooth muscle|myocyte|sarcomere)\b',
            # Sensory
            r'\b(?:eye|cornea|lens|retina|ear|cochlea|vestibular|taste bud|olfactory)\b',
            # Skeletal
            r'\b(?:bone|cartilage|osteoblast|osteoclast|chondrocyte|joint|ligament|tendon)\b'
        ]
        
        # Filter records with primary tissue samples in memory
        tissue_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        
        text_columns = ['overall_design', 'characteristics_ch1', 'summary']
        
        for pattern in tissue_patterns:
            for col in text_columns:
                if col in input_result.data.columns:
                    tissue_mask |= input_result.data[col].str.contains(pattern, case=False, na=False, regex=True)
        
        filtered_df = input_result.data[tissue_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Tissue Source",
            description="Primary tissue samples"
        )
        
        result.log_result(input_result.count)
        return result

class KeywordSearchFilter(BaseFilter):
    """
    Keyword search filter class.
    """
    
    def __init__(self, conn: connection, search_term: Optional[str]):
        super().__init__(conn)
        self.search_term = search_term
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Filter records based on keywords.
        """
        if input_result.data.empty or not self.search_term or not self.search_term.strip():
            return input_result
        
        # Search records in memory
        search_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        search_columns = ['study_title', 'summary', 'gse_title']
        
        for col in search_columns:
            if col in input_result.data.columns:
                search_mask |= input_result.data[col].str.contains(
                    self.search_term, case=False, na=False, regex=False
                )
        
        filtered_df = input_result.data[search_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Keyword Search",
            description=f"Contains '{self.search_term}'"
        )
        
        result.log_result(input_result.count)
        return result

class LimitFilter(BaseFilter):
    """
    Limit filter class.
    """
    
    def __init__(self, conn: connection, limit: int):
        super().__init__(conn)
        self.limit = limit
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """
        Limit the number of records returned.
        """
        if input_result.data.empty or self.limit <= 0:
            return input_result
        
        # Sort by submission date and limit the number of records
        sorted_df = input_result.data.copy()
        if 'gsm_submission_date' in sorted_df.columns:
            sorted_df = sorted_df.sort_values('gsm_submission_date', ascending=False, na_position='last')
        
        limited_df = sorted_df.head(self.limit)
        
        result = FilterResult(
            data=limited_df,
            count=len(limited_df),
            filter_name="Limit Filter",
            description=f"Top {self.limit} records by submission date"
        )
        
        result.log_result(input_result.count)
        return result

# Convenience function: create filter chain
def create_filter_chain(conn: connection, 
                       organisms: List[str] = ["human"],
                       search_term: Optional[str] = None,
                       limit: int = 100,
                       min_sc_confidence: int = 1,
                       include_initial_dataset: bool = True,
                       include_basic_availability: bool = True,
                       include_organism: bool = True,
                       include_single_cell: bool = True,
                       include_tissue_source: bool = True,
                       include_sequencing_strategy: bool = False,
                       include_cancer_status: bool = False,
                       include_search_term: bool = False,
                       include_exclusion_single_cell: bool = False) -> List[BaseFilter]:
    """
    Create a prefilter chain.
    
    Args:
        conn: Database connection
        organisms: List of organisms, default ["human"]
        search_term: Search keyword
        limit: Limit on the number of records returned
        min_sc_confidence: Minimum single-cell confidence score
        include_initial_dataset: Whether to include initial dataset filter
        include_basic_availability: Whether to include basic availability filter
        include_organism: Whether to include organism filter
        include_single_cell: Whether to include single cell filter
        include_tissue_source: Whether to include tissue source filter
        include_sequencing_strategy: Whether to include sequencing strategy filter
        include_cancer_status: Whether to include cancer status filter
        include_search_term: Whether to include keyword search filter
        include_exclusion_single_cell: Whether to include exclusion single cell filter
    
    Returns:
        List of filter objects
    """
    # Create base filters based on parameters
    filters = []
    
    if include_initial_dataset:
        filters.append(InitialDatasetFilter(conn))
    
    if include_basic_availability:
        filters.append(BasicAvailabilityFilter(conn))
    
    if include_organism:
        filters.append(OrganismFilter(conn, organisms))
    
    if include_single_cell:
        filters.append(SingleCellFilter(conn, min_sc_confidence))

    if include_exclusion_single_cell:
        filters.append(ExclusionSingleCellFilter(conn))
    
    if include_tissue_source:
        filters.append(TissueSourceFilter(conn))
    
    # Add optional filters based on parameters
    if include_sequencing_strategy:
        filters.append(SequencingStrategyFilter(conn))
    
    if include_cancer_status:
        filters.append(CancerStatusFilter(conn))
    
    if include_search_term and search_term:
        filters.append(KeywordSearchFilter(conn, search_term))
    
    # Always add limit filter at the end
    filters.append(LimitFilter(conn, limit))
    
    return filters

def apply_filter_chain(filter_chain: List[BaseFilter]) -> FilterResult:
    """
    Apply a prefilter chain.
    """
    print("🔍 Starting Stepwise Prefiltering Process")
    print("=" * 50)
    
    result = None
    for filter_obj in filter_chain:
        result = filter_obj.apply(result)
        if result.count == 0:
            print("⚠️  No records remaining after this filter. Stopping chain.")
            break
    
    print("=" * 50)
    print(f"📈 Final result: {result.count:,} records")
    return result

class FilterChainManager:
    """
    Manages filter cchains with reporting and debugging
    """

    def __init__(self, conn: connection):
        self.conn = conn
        self.execution_history: List[Dict[str, Any]] = []
    
    def create_standard_chain(self, 
                            organisms: List[str] = ["human"],
                            search_term: Optional[str] = None,
                            limit: int = 100,
                            min_sc_confidence: int = 1,
                            use_sc_fallback: bool = True,
                            include_initial_dataset: bool = True,
                            include_basic_availability: bool = True,
                            include_organism: bool = True,
                            include_single_cell: bool = True,
                            include_tissue_source: bool = True,
                            include_sequencing_strategy: bool = False,
                            include_cancer_status: bool = False,
                            include_search_term: bool = False,
                            include_exclusion_single_cell: bool = False) -> List[BaseFilter]:
        """
        Create a standard filter chain with enhanced options.
        
        Args:
            organisms: List of organisms, default ["human"]
            search_term: Search keyword
            limit: Limit on the number of records returned
            min_sc_confidence: Minimum single-cell confidence score
            use_sc_fallback: Whether to use fallback strategy for single cell detection
            include_initial_dataset: Whether to include initial dataset filter
            include_basic_availability: Whether to include basic availability filter
            include_organism: Whether to include organism filter
            include_single_cell: Whether to include single cell filter
            include_tissue_source: Whether to include tissue source filter
            include_sequencing_strategy: Whether to include sequencing strategy filter
            include_cancer_status: Whether to include cancer status filter
            include_search_term: Whether to include keyword search filter
        
        Returns:
            List of filter objects
        """
        filters = []
        
        # Add filters based on parameters
        if include_initial_dataset:
            filters.append(InitialDatasetFilter(self.conn))
        
        if include_basic_availability:
            filters.append(BasicAvailabilityFilter(self.conn))
        
        if include_organism and organisms:
            filters.append(OrganismFilter(self.conn, organisms))
        
        if include_single_cell:
            filters.append(SingleCellFilter(self.conn, min_sc_confidence, use_sc_fallback))

        if include_exclusion_single_cell:
            filters.append(ExclusionSingleCellFilter(self.conn))
        
        if include_tissue_source:
            filters.append(TissueSourceFilter(self.conn))
        
        if include_sequencing_strategy:
            filters.append(SequencingStrategyFilter(self.conn))
        
        if include_cancer_status:
            filters.append(CancerStatusFilter(self.conn))
        
        if include_search_term and search_term:
            filters.append(KeywordSearchFilter(self.conn, search_term))
        
        # Always add limit filter at the end if specified
        if limit > 0:
            filters.append(LimitFilter(self.conn, limit))
        
        return filters
    
    def apply_chain(self, filter_chain: List[BaseFilter], 
                   stop_on_empty: bool = True,
                   show_summary: bool = True) -> FilterResult:
        """
        Apply filter chain with enhanced reporting.
        """
        print("🔍 Starting Enhanced Stepwise Prefiltering Process")
        print("=" * 60)
        
        start_time = datetime.now()
        result = None
        chain_history = []
        
        for i, filter_obj in enumerate(filter_chain, 1):
            print(f"Step {i}: {filter_obj.name}")
            print("-" * 40)
            
            step_start = datetime.now()
            result = filter_obj.apply(result)
            step_end = datetime.now()
            
            step_summary = result.get_summary()
            step_summary['step_number'] = i
            step_summary['step_duration'] = (step_end - step_start).total_seconds()
            chain_history.append(step_summary)
            
            if result.count == 0 and stop_on_empty:
                print("⚠️  No records remaining after this filter. Stopping chain.")
                break
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Store execution history
        execution_record = {
            'timestamp': start_time.isoformat(),
            'total_duration': total_time,
            'final_count': result.count if result else 0,
            'chain_history': chain_history
        }
        self.execution_history.append(execution_record)
        
        if show_summary:
            self._print_summary(chain_history, total_time, result)
        
        return result
    
    def _print_summary(self, chain_history: List[Dict], total_time: float, final_result: FilterResult):
        """
        Print execution summary.
        """
        print("=" * 60)
        print(f"📈 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Final result: {final_result.count:,} records")
        print(f"Total reduction: {chain_history[0]['count'] - final_result.count:,} records")
        
        # Show step-by-step breakdown
        print("\n📊 Step-by-step breakdown:")
        for step in chain_history:
            print(f"  {step['step_number']:2d}. {step['filter_name']:<20} "
                  f"{step['count']:>6,} records "
                  f"({step['reduction_pct']:>5.1f}% reduction, "
                  f"{step['step_duration']:>5.2f}s)")
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about filter chain executions.
        """
        if not self.execution_history:
            return {}
        
        recent_executions = self.execution_history[-10:]  # Last 10 executions
        
        avg_time = sum(exec['total_duration'] for exec in recent_executions) / len(recent_executions)
        avg_final_count = sum(exec['final_count'] for exec in recent_executions) / len(recent_executions)
        
        return {
            'total_executions': len(self.execution_history),
            'avg_execution_time': avg_time,
            'avg_final_count': avg_final_count,
            'last_execution': self.execution_history[-1] if self.execution_history else None
        }

# Usage example:
def example_enhanced_usage():
    """
    Example of using the enhanced prefilter system.
    """
    
    # Initialize chain manager
    manager = FilterChainManager(conn)
    
    # Create enhanced chain with fallback options
    filter_chain = manager.create_standard_chain(
        organisms=["human"],
        search_term="cancer",
        limit=50,
        min_sc_confidence=1,  # Lower threshold
        use_sc_fallback=True,  # Enable text-based fallback
        include_exclusion_single_cell=True # Enable exclusion-based single cell filter
    )
    
    # Apply with enhanced reporting
    result = manager.apply_chain(
        filter_chain, 
        stop_on_empty=False,  # Continue even if a filter returns 0 records
        show_summary=True
    )
    
    # Get execution statistics
    stats = manager.get_execution_stats()
    print(f"\n📊 Execution Statistics: {stats}")
    
    return result