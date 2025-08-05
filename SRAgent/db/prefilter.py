import warnings
import pandas as pd
import logging
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
                'sra_ID', 'study_title', 'summary', 'overall_design', 'scientific_name',
                'library_strategy', 'technology', 'characteristics_ch1', 'gse_title', 
                'gsm_title', 'organism_ch1', 'source_name_ch1', 'common_name', 
                'gsm_submission_date', 'sc_conf_score', 'study_alias'
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
        human_mask = (
            input_result.data['organism_ch1'].str.contains('homo sapiens', case=False, na=False) |
            input_result.data['scientific_name'].str.contains('homo sapiens', case=False, na=False) |
            input_result.data['source_name_ch1'].str.contains('human', case=False, na=False) |
            input_result.data['common_name'].str.contains('human', case=False, na=False)
        )
        
        filtered_df = input_result.data[human_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Organism Filter",
            description="Human samples only"
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
        Filter samples from major tissue systems, excluding cell lines.
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
        
        # Exclude cell line keywords
        cell_line_patterns = [
            r'\b(?:cell line|immortalized|HEK293|HeLa|Jurkat|CHO|NIH/3T3|K562|U2OS|HEPG2|A549)\b'
        ]
        
        # Filter records with primary tissue samples in memory
        tissue_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        cell_line_mask = pd.Series([False] * len(input_result.data), index=input_result.data.index)
        
        text_columns = ['overall_design', 'characteristics_ch1', 'summary']
        
        for pattern in tissue_patterns:
            for col in text_columns:
                if col in input_result.data.columns:
                    tissue_mask |= input_result.data[col].str.contains(pattern, case=False, na=False, regex=True)
        
        for pattern in cell_line_patterns:
            for col in text_columns:
                if col in input_result.data.columns:
                    cell_line_mask |= input_result.data[col].str.contains(pattern, case=False, na=False, regex=True)
        
        final_mask = tissue_mask & ~cell_line_mask
        filtered_df = input_result.data[final_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Tissue Source",
            description="Primary tissue samples (not cell lines)"
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
                       min_sc_confidence: int = 2,
                       include_sequencing_strategy: bool = False,
                       include_cancer_status: bool = False,
                       include_search_term: bool = False) -> List[BaseFilter]:
    """
    Create a prefilter chain.
    
    Args:
        conn: Database connection
        organisms: List of organisms, default ["human"]
        search_term: Search keyword
        limit: Limit on the number of records returned
        min_sc_confidence: Minimum single-cell confidence score
        include_sequencing_strategy: Whether to include sequencing strategy filter
        include_cancer_status: Whether to include cancer status filter
        include_search_term: Whether to include keyword search filter
    
    Returns:
        List of filter objects
    """
    # Create base filters that are always included
    filters = [
        InitialDatasetFilter(conn),
        BasicAvailabilityFilter(conn),
        OrganismFilter(conn, organisms),
        SingleCellFilter(conn, min_sc_confidence),
        TissueSourceFilter(conn)
    ]
    
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
                            min_sc_confidence: int = 2,
                            use_sc_fallback: bool = True,
                            include_sequencing_strategy: bool = False,
                            include_cancer_status: bool = False,
                            include_search_term: bool = False) -> List[BaseFilter]:
        """
        Create a standard filter chain with enhanced options.
        
        Args:
            organisms: List of organisms, default ["human"]
            search_term: Search keyword
            limit: Limit on the number of records returned
            min_sc_confidence: Minimum single-cell confidence score
            use_sc_fallback: Whether to use fallback strategy for single cell detection
            include_sequencing_strategy: Whether to include sequencing strategy filter
            include_cancer_status: Whether to include cancer status filter
            include_search_term: Whether to include keyword search filter
        
        Returns:
            List of filter objects
        """
        # Create base filters that are always included
        filters = [
            InitialDatasetFilter(self.conn),
            BasicAvailabilityFilter(self.conn),
            OrganismFilter(self.conn, organisms),
            SingleCellFilter(self.conn, min_sc_confidence, use_sc_fallback),
            TissueSourceFilter(self.conn)
        ]
        
        # Add optional filters based on parameters
        if include_sequencing_strategy:
            filters.append(SequencingStrategyFilter(self.conn))
        
        if include_cancer_status:
            filters.append(CancerStatusFilter(self.conn))
        
        if include_search_term and search_term:
            filters.append(KeywordSearchFilter(self.conn, search_term))
        
        # Always add limit filter at the end
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
        use_sc_fallback=True  # Enable text-based fallback
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