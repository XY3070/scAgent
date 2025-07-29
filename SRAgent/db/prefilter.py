#!/usr/bin/env python3
"""
db/prefilter.py

独立的预筛选函数模块，每个函数接受一个数据集对象，返回筛选后的新对象
采用函数式编程思想，环环相扣，逐步筛选
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Set
from psycopg2.extensions import connection
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class FilterResult:
    """筛选结果数据类"""
    data: pd.DataFrame
    count: int
    filter_name: str
    description: str
    reduction_count: int = 0
    reduction_pct: float = 0.0
    
    def __post_init__(self):
        """计算记录数"""
        if self.data is not None:
            self.count = len(self.data)
    
    def log_result(self, previous_count: int = None):
        """记录筛选结果"""
        if previous_count is not None:
            self.reduction_count = previous_count - self.count
            self.reduction_pct = (self.reduction_count / previous_count * 100) if previous_count > 0 else 0
            
        print(f"📊 {self.filter_name}: {self.count:,} records "
              f"(↓{self.reduction_count:,}, -{self.reduction_pct:.1f}%)")
        if self.description:
            print(f"   {self.description}")
        print()

class BaseFilter(ABC):
    """预筛选器基类"""
    
    def __init__(self, conn: connection):
        self.conn = conn
        
    @abstractmethod
    def apply(self, input_result: FilterResult) -> FilterResult:
        """应用筛选器"""
        pass
    
    def execute_query_with_cursor(self, query: str, params: tuple) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
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
    """初始数据集筛选器"""
    
    def __init__(self, conn: connection, table_name: str = "merged.sra_geo_ft"):
        super().__init__(conn)
        self.table_name = table_name
    
    def apply(self, input_result: FilterResult = None) -> FilterResult:
        """获取初始数据集"""
        query = f"""
            SELECT "sra_ID", study_title, summary, overall_design, scientific_name, 
                   library_strategy, technology, characteristics_ch1, gse_title, gsm_title,
                   organism_ch1, source_name_ch1, common_name, gsm_submission_date, sc_conf_score
            FROM {self.table_name}
        """
        
        df = self.execute_query_with_cursor(query, ())
        
        result = FilterResult(
            data=df,
            count=len(df),
            filter_name="Initial Dataset",
            description="All records in database"
        )
        
        result.log_result()
        return result

class BasicAvailabilityFilter(BaseFilter):
    """基础数据可用性筛选器"""
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """筛选具有基础数据的记录"""
        if input_result.data.empty:
            return FilterResult(
                data=pd.DataFrame(),
                count=0,
                filter_name="Basic Availability",
                description="No input data"
            )
        
        # 在内存中筛选，而不是重新查询数据库
        filtered_df = input_result.data[
            (input_result.data['sra_ID'].notna()) & 
            (input_result.data['sra_ID'] != '') & 
            (input_result.data['gse_title'].notna())
        ].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Basic Availability",
            description="Has SRA_ID and GSE title"
        )
        
        result.log_result(input_result.count)
        return result

class OrganismFilter(BaseFilter):
    """物种筛选器"""
    
    def __init__(self, conn: connection, organisms: List[str]):
        super().__init__(conn)
        self.organisms = organisms
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """筛选指定物种的记录"""
        if input_result.data.empty or not self.organisms:
            return input_result
            
        if "human" not in [org.lower() for org in self.organisms]:
            return input_result
        
        # 在内存中筛选
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
    """单细胞筛选器"""
    
    def __init__(self, conn: connection, min_confidence: int = 2):
        super().__init__(conn)
        self.min_confidence = min_confidence
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """筛选单细胞数据"""
        if input_result.data.empty:
            return input_result
        
        # 检查sc_conf_score字段是否存在
        if 'sc_conf_score' not in input_result.data.columns:
            logger.warning("sc_conf_score column not found, skipping single cell filter")
            return input_result
        
        # 在内存中筛选
        sc_mask = (
            input_result.data['sc_conf_score'].notna() & 
            (input_result.data['sc_conf_score'] >= self.min_confidence)
        )
        
        filtered_df = input_result.data[sc_mask].copy()
        
        result = FilterResult(
            data=filtered_df,
            count=len(filtered_df),
            filter_name="Single Cell Filter",
            description=f"Single cell confidence score >= {self.min_confidence}"
        )
        
        result.log_result(input_result.count)
        return result

class SequencingStrategyFilter(BaseFilter):
    """测序策略筛选器"""
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """筛选单细胞测序技术"""
        if input_result.data.empty:
            return input_result
        
        # 定义单细胞测序技术的关键词
        sc_tech_patterns = [
            r'10[xX]', r'\b10x\b', 'Chromium', 'SMART-seq2', 'Drop-seq', 
            'microwell', 'C1\s?System', 'Single cell sequencing', 'scRNA-seq', 
            'snRNA-seq', 'CITE-seq'
        ]
        
        # 排除的关键词
        exclude_patterns = ['bulk', 'microorganism', 'yeast', 'E\. coli', 'bulk RNA', 'tissue profiling']
        
        # 在内存中筛选
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
    """癌症状态筛选器"""
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """筛选癌症或正常样本"""
        if input_result.data.empty:
            return input_result
        
        # 癌症相关关键词
        cancer_patterns = [
            'tumor', 'cancer', 'carcinoma', 'neoplasm', 'malignant', 'metastatic', 
            'onco', 'glioma', 'sarcoma', 'leukemia', 'lymphoma', 'adenocarcinoma', 
            'melanoma', 'osteosarcoma', 'neuroblastoma'
        ]
        
        # 正常样本关键词
        normal_patterns = [
            'normal', 'non-?tumor', 'healthy', 'control', 'donor', 'peripheral blood', 
            'adjacent normal', 'para-tumor', 'healthy donor'
        ]
        
        # 在内存中筛选
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
    """组织来源筛选器"""
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """筛选来自主要组织系统的样本，排除细胞系"""
        if input_result.data.empty:
            return input_result
        
        # 主要组织系统关键词
        tissue_patterns = [
            # Neural
            r'\b(brain|cerebral|hippocampus|cortex|striatum|cerebellum|spinal cord|retina|optic nerve|glia|neuron)\b',
            # Respiratory
            r'\b(lung|pulmonary|alveolar|bronchial|trachea|nasal|larynx|pharynx|diaphragm)\b',
            # Digestive
            r'\b(liver|hepatic|gastric|stomach|intestine|colon|ileum|jejunum|duodenum|esophagus|pancreas|gallbladder|bile duct)\b',
            # Circulatory
            r'\b(heart|cardiac|myocardial|vascular|artery|vein|capillary|blood|pbmc|plasma|serum)\b',
            # Immune
            r'\b(lymph node|spleen|thymus|bone marrow|tonsil|macrophage|lymphocyte|neutrophil|dendritic cell|mast cell)\b',
            # Urinary
            r'\b(kidney|renal|nephron|bladder|ureter|urethra)\b',
            # Reproductive
            r'\b(ovary|testis|uterus|placenta|prostate|penis|vagina|fallopian tube|endometrium)\b',
            # Endocrine
            r'\b(thyroid|parathyroid|adrenal|pituitary|pancreatic islet|hypothalamus)\b',
            # Skin
            r'\b(skin|epidermis|dermis|melanocyte|hair follicle|sweat gland|sebaceous gland)\b',
            # Muscle
            r'\b(muscle|skeletal muscle|cardiac muscle|smooth muscle|myocyte|sarcomere)\b',
            # Sensory
            r'\b(eye|cornea|lens|retina|ear|cochlea|vestibular|taste bud|olfactory)\b',
            # Skeletal
            r'\b(bone|cartilage|osteoblast|osteoclast|chondrocyte|joint|ligament|tendon)\b'
        ]
        
        # 细胞系排除关键词
        cell_line_patterns = [
            r'\b(cell line|immortalized|HEK293|HeLa|Jurkat|CHO|NIH/3T3|K562|U2OS|HEPG2|A549)\b'
        ]
        
        # 在内存中筛选
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
    """关键词搜索筛选器"""
    
    def __init__(self, conn: connection, search_term: Optional[str]):
        super().__init__(conn)
        self.search_term = search_term
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """根据关键词筛选"""
        if input_result.data.empty or not self.search_term or not self.search_term.strip():
            return input_result
        
        # 在内存中搜索
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
    """记录数限制筛选器"""
    
    def __init__(self, conn: connection, limit: int):
        super().__init__(conn)
        self.limit = limit
    
    def apply(self, input_result: FilterResult) -> FilterResult:
        """限制返回记录数"""
        if input_result.data.empty or self.limit <= 0:
            return input_result
        
        # 按提交日期排序并限制记录数
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

# 便利函数：创建筛选器链
def create_filter_chain(conn: connection, 
                       organisms: List[str] = ["human"],
                       search_term: Optional[str] = None,
                       limit: int = 100,
                       min_sc_confidence: int = 2) -> List[BaseFilter]:
    """创建预筛选器链"""
    return [
        InitialDatasetFilter(conn),
        BasicAvailabilityFilter(conn),
        OrganismFilter(conn, organisms),
        SingleCellFilter(conn, min_sc_confidence),
        SequencingStrategyFilter(conn),
        CancerStatusFilter(conn),
        TissueSourceFilter(conn),
        KeywordSearchFilter(conn, search_term),
        LimitFilter(conn, limit)
    ]

def apply_filter_chain(filter_chain: List[BaseFilter]) -> FilterResult:
    """应用筛选器链"""
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