#!/usr/bin/env python3
"""
db/get.py

更新后的数据库查询函数，使用预筛选模块
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Set
import psycopg2 
import pandas as pd
from pypika import Query, Table, Field, Column, Criterion
from psycopg2.extras import execute_values
from psycopg2.extensions import connection
import logging

# 导入预筛选模块
from .prefilter import (
    FilterResult, 
    create_filter_chain, 
    apply_filter_chain,
    InitialDatasetFilter,
    BasicAvailabilityFilter,
    OrganismFilter,
    SingleCellFilter,
    SequencingStrategyFilter,
    CancerStatusFilter,
    TissueSourceFilter,
    KeywordSearchFilter,
    LimitFilter
)

# 原有的数据库工具函数保持不变
from .utils import execute_query

logger = logging.getLogger(__name__)

def execute_query_with_cursor(conn, query, params):
    """
    修复版本：正确处理数据库查询和结果
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        if cursor.description is None:
            cursor.close()
            return pd.DataFrame()
        
        colnames = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        cursor.close()
        
        return pd.DataFrame(results, columns=colnames)
        
    except Exception as e:
        print(f"Database query error: {e}")
        try:
            cursor.close()
        except:
            pass
        return pd.DataFrame()

# 保留所有原有的函数
def db_find_srx(srx_accessions: List[str], conn: connection) -> pd.DataFrame:
    """
    Get SRX records on the database
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    srx_metadata = Table("srx_metadata")
    stmt = Query \
        .from_(srx_metadata) \
        .select("*") \
        .distinct() \
        .where(srx_metadata.srx_accession.isin(srx_accessions))
    # return as pandas dataframe
    df = pd.read_sql(str(stmt), conn)
    return df

def db_get_srx_records(conn: connection, column: str="entrez_id", database: str="sra") -> List[int]:
    """
    Get the entrez_id values of all SRX records in the database.
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    srx_metadata = Table("srx_metadata")
    target_column = getattr(srx_metadata, column)
    stmt = Query \
        .from_(srx_metadata) \
        .select(target_column) \
        .distinct() \
        .where(srx_metadata.database == database)
        
    # Fetch the results and return a list of {target_column} values
    results = execute_query(stmt, conn)
    return [row[0] for row in results] if results else []

def db_get_unprocessed_records(conn: connection, database: str="sra") -> List[int]:
    """
    Get the entrez_id values of SRX records that have not been processed.
    Args:
        conn: Connection to the database.
        database: Name of the database to query.
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    srx_metadata = Table("srx_metadata")
    srx_srr = Table("srx_srr")
    stmt = Query \
        .from_(srx_metadata) \
        .left_join(srx_srr) \
        .on(srx_metadata.srx_accession == srx_srr.srx_accession) \
        .select(srx_metadata.entrez_id) \
        .distinct() \
        .where(
            Criterion.all([
                srx_metadata.database == database,
                srx_srr.srr_accession.isnull()
            ])
        )
        
    # Fetch the results and return a list of entrez_id values
    results = execute_query(stmt, conn)
    return [row[0] for row in results] if results else []

def db_get_filtered_srx_metadata(
    conn: connection, 
    organism: str = None,
    is_single_cell: str = None,
    query: str = None,
    limit: int = 100,
    database: str="sra"
    ) -> pd.DataFrame:
    """
    Get filtered SRX metadata records from the database.
    Args:
        conn: Connection to the database.
        organism: Organism to filter by (e.g., "Homo sapiens").
        is_single_cell: Filter by single cell status ("yes" or "no").
        limit: Maximum number of records to return.
        database: Name of the database to query.
    Returns:
        dataframe of filtered SRX metadata records.
    """
    srx_metadata = Table("srx_metadata")

    criteria = [srx_metadata.database == database]
    if organism:
        criteria.append(srx_metadata.organism == organism)
    if is_single_cell:
        criteria.append(srx_metadata.is_single_cell == is_single_cell)
    if query:
        # Add a case-insensitive search across multiple text fields
        # Using ilike for case-insensitive LIKE in PostgreSQL
        search_pattern = f"%{query.lower()}%"
        criteria.append(
            Criterion.any([
                srx_metadata.title.ilike(search_pattern),
                srx_metadata.design_description.ilike(search_pattern)
            ])
        )

    stmt = Query \
        .from_(srx_metadata) \
        .select(
            srx_metadata.srx_accession,
            srx_metadata.entrez_id,
            srx_metadata.organism,
            srx_metadata.is_single_cell,
            srx_metadata.tech_10x,
            srx_metadata.library_strategy,
            srx_metadata.library_source,
            srx_metadata.library_selection,
            srx_metadata.platform,
            srx_metadata.instrument_model,
            srx_metadata.sra_study_accession,
            srx_metadata.bioproject_accession,
            srx_metadata.biosample_accession,
            srx_metadata.pubmed_id,
            srx_metadata.title,
            srx_metadata.design_description,
            srx_metadata.sample_description,
            srx_metadata.submission_date,
            srx_metadata.update_date
        ) \
        .where(Criterion.all(criteria)) \
        .limit(limit)
        
    # fetch as pandas dataframe
    df = pd.read_sql(str(stmt), conn)
    return df if not df.empty else []

def db_get_srx_accessions(
    conn: connection, database: str="sra"
    ) -> Set[int]:
    """
    Get all SRX accessions in the screcounter database
    Args:
        conn: Connection to the database.
        database: Name of the sequence database (e.g., sra)
    Returns:
        Set of SRX accessions in the database.
    """
    srx_metadata = Table("srx_metadata")
    stmt = Query \
        .from_(srx_metadata) \
        .where(
            srx_metadata.database == database
        ) \
        .select(
            srx_metadata.srx_accession
        ) \
        .distinct()
        
    # fetch records
    with conn.cursor() as cur:
        cur.execute(str(stmt))
        results = cur.fetchall()
        return set([int(row[0]) for row in results]) if results else set()

def db_get_entrez_ids(
    conn: connection, database: str="sra"
    ) -> Set[int]:
    """
    Get all Entrez IDs in the screcounter database
    Args:
        conn: Connection to the database.
        database: Name of the sequence database (e.g., sra)
    Returns:
        Set of Entrez IDs in the database. Returns empty set if no results.
    """
    srx_metadata = Table("srx_metadata")
    stmt = Query \
        .from_(srx_metadata) \
        .where(
            srx_metadata.database == database
        ) \
        .select(
            srx_metadata.entrez_id
        ) \
        .distinct()
        
    # fetch records
    with conn.cursor() as cur:
        cur.execute(str(stmt))
        results = cur.fetchall()
        return set([int(row[0]) for row in results]) if results else set()

def db_get_eval(conn: connection, dataset_ids: List[str]) -> pd.DataFrame:
    """
    Get the entrez_id values of all SRX records in the database.
    Args:
        conn: Connection to the database.
        dataset_ids: List of dataset_ids to return
    Returns:
        List of entrez_id values of SRX records that have not been processed.
    """
    tbl = Table("eval")
    stmt = Query \
        .from_(tbl) \
        .select(tbl.dataset_id) \
        .distinct() \
        .where(tbl.dataset_id.isin(dataset_ids))
        
    # Fetch the results and return a list of {target_column} values
    return [row[0] for row in execute_query(stmt, conn)]

def db_get_table_data(conn: connection, table_name: str) -> pd.DataFrame:
    """
    Get all data from a specified table.
    Args:
        conn: Connection to the database.
        table_name: The name of the table to query.
    Returns:
        DataFrame containing all data from the specified table.
    """
    tbl = Table(table_name)
    stmt = Query \
        .from_(tbl) \
        .select("*")
    return pd.read_sql(str(stmt), conn)

# 新的预筛选函数，使用独立的筛选器模块
def get_prefiltered_datasets_functional(
    conn: connection,
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 100,
    min_sc_confidence: int = 2,
    create_temp_table: bool = False,
    temp_table_name: str = "temp_prefiltered_results"
) -> pd.DataFrame:
    """
    使用函数式预筛选方法，每个筛选器接受一个对象，返回新的筛选后对象
    
    Args:
        conn: 数据库连接
        organisms: 物种列表，默认["human"]
        search_term: 搜索关键词
        limit: 返回记录数限制
        min_sc_confidence: 单细胞置信度最小值
        create_temp_table: 是否创建临时表
        temp_table_name: 临时表名称
    
    Returns:
        预筛选后的DataFrame
    """
    try:
        # 创建筛选器链
        filter_chain = create_filter_chain(
            conn=conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit,
            min_sc_confidence=min_sc_confidence
        )
        
        # 应用筛选器链
        final_result = apply_filter_chain(filter_chain)
        
        # 如果需要创建临时表
        if create_temp_table and not final_result.data.empty:
            create_temporary_table(conn, final_result.data, temp_table_name)
        
        return final_result.data
        
    except Exception as e:
        logger.error(f"Prefiltering failed: {e}")
        return pd.DataFrame()

def get_prefiltered_datasets_custom_chain(
    conn: connection,
    custom_filters: List[str],
    filter_params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    使用自定义筛选器链进行预筛选
    
    Args:
        conn: 数据库连接
        custom_filters: 自定义筛选器名称列表
        filter_params: 筛选器参数字典
    
    Returns:
        预筛选后的DataFrame
    """
    if filter_params is None:
        filter_params = {}
    
    # 映射筛选器名称到类
    filter_map = {
        'initial': InitialDatasetFilter,
        'basic': BasicAvailabilityFilter,
        'organism': OrganismFilter,
        'single_cell': SingleCellFilter,
        'sequencing': SequencingStrategyFilter,
        'cancer': CancerStatusFilter,
        'tissue': TissueSourceFilter,
        'keyword': KeywordSearchFilter,
        'limit': LimitFilter
    }
    
    try:
        # 构建自定义筛选器链
        filter_chain = []
        result = None
        
        for filter_name in custom_filters:
            if filter_name not in filter_map:
                logger.warning(f"Unknown filter: {filter_name}")
                continue
            
            filter_class = filter_map[filter_name]
            
            # 根据筛选器类型创建实例
            if filter_name == 'organism':
                filter_obj = filter_class(conn, filter_params.get('organisms', ['human']))
            elif filter_name == 'single_cell':
                filter_obj = filter_class(conn, filter_params.get('min_sc_confidence', 2))
            elif filter_name == 'keyword':
                filter_obj = filter_class(conn, filter_params.get('search_term'))
            elif filter_name == 'limit':
                filter_obj = filter_class(conn, filter_params.get('limit', 100))
            else:
                filter_obj = filter_class(conn)
            
            # 应用筛选器
            result = filter_obj.apply(result)
            
            if result.count == 0:
                logger.warning("No records remaining after filter: " + filter_name)
                break
        
        return result.data if result else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Custom chain filtering failed: {e}")
        return pd.DataFrame()

def create_temporary_table(conn: connection, df: pd.DataFrame, table_name: str):
    """
    创建临时表并插入预筛选结果
    
    Args:
        conn: 数据库连接
        df: 要插入的DataFrame
        table_name: 临时表名称
    """
    try:
        cursor = conn.cursor()
        
        # 删除已存在的临时表
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # 创建临时表结构（基于DataFrame的列）
        columns_def = []
        for col in df.columns:
            # 简单的类型映射，可以根据需要扩展
            if df[col].dtype == 'object':
                columns_def.append(f'"{col}" TEXT')
            elif df[col].dtype == 'int64':
                columns_def.append(f'"{col}" INTEGER')
            elif df[col].dtype == 'float64':
                columns_def.append(f'"{col}" REAL')
            else:
                columns_def.append(f'"{col}" TEXT')
        
        create_sql = f"CREATE TEMP TABLE {table_name} ({', '.join(columns_def)})"
        cursor.execute(create_sql)
        
        # 插入数据
        if not df.empty:
            # 准备插入语句
            placeholders = ', '.join(['%s'] * len(df.columns))
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
            
            # 转换DataFrame为元组列表
            data_tuples = [tuple(row) for row in df.values]
            
            # 批量插入
            cursor.executemany(insert_sql, data_tuples)
        
        conn.commit()
        cursor.close()
        
        logger.info(f"Created temporary table '{table_name}' with {len(df)} records")
        
    except Exception as e:
        logger.error(f"Failed to create temporary table: {e}")
        try:
            cursor.close()
        except:
            pass

# 保留原始函数的兼容性版本
async def get_prefiltered_datasets_from_local_db(
    conn,
    organisms: list,
    search_term: str,
    limit: int = 100
) -> list:
    """
    原始预筛选函数的兼容性版本，现在使用新的函数式筛选器
    保持与原有代码的兼容性
    """
    try:
        # 调用新的函数式预筛选方法
        result_df = get_prefiltered_datasets_functional(
            conn=conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit
        )
        
        if result_df.empty:
            logger.info("No records found with current filters")
            return []
        
        # 转换为原始格式（字典列表）
        records = result_df.to_dict(orient='records')
        logger.info(f"Successfully found {len(records)} records")
        return records
        
    except Exception as e:
        logger.error(f"Prefiltering error: {e}")
        return []

def check_table_structure(conn):
    """
    Check the table structure and confirm which fields actually exist
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'merged' AND table_name = 'sra_geo_ft'
            ORDER BY column_name
        """)
        
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        print("📊 Actual fields existing in the database table:")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        
        return columns
        
    except Exception as e:
        print(f"❌ Failed to check table structure: {e}")
        return []

# 示例用法函数
def example_usage():
    """展示如何使用新的预筛选系统"""
    from dotenv import load_dotenv
    from .connect import db_connect
    
    load_dotenv()
    
    with db_connect() as conn:
        print("=== 示例1: 标准预筛选流程 ===")
        result1 = get_prefiltered_datasets_functional(
            conn=conn,
            organisms=["human"],
            search_term="cancer",
            limit=10
        )
        print(f"结果1: {len(result1)} 条记录\n")
        
        print("=== 示例2: 自定义筛选器链 ===")
        result2 = get_prefiltered_datasets_custom_chain(
            conn=conn,
            custom_filters=['initial', 'basic', 'organism', 'keyword', 'limit'],
            filter_params={
                'organisms': ['human'],
                'search_term': 'brain',
                'limit': 5
            }
        )
        print(f"结果2: {len(result2)} 条记录\n")
        
        print("=== 示例3: 逐步手动筛选 ===")
        # 手动创建筛选器链，可以在任意步骤停止或修改
        initial_filter = InitialDatasetFilter(conn)
        basic_filter = BasicAvailabilityFilter(conn)
        organism_filter = OrganismFilter(conn, ["human"])
        sc_filter = SingleCellFilter(conn, min_confidence=2)
        
        # 逐步应用
        result = initial_filter.apply()
        result = basic_filter.apply(result)
        result = organism_filter.apply(result)
        result = sc_filter.apply(result)
        
        print(f"结果3: {result.count} 条记录")

# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from SRAgent.db.connect import db_connect
    
    os.environ["DYNACONF"] = "test"
    with db_connect() as conn:
        # 运行示例
        example_usage()
        
        # 原有的测试代码
        print(db_get_eval(conn, ["eval1"]))
        print(db_get_srx_records(conn))
        print(db_get_unprocessed_records(conn))
        print(len(db_get_srx_accessions(conn)))
        print(db_find_srx(["SRX19162973"], conn))
        
        # Example usage for the new function
        metadata = db_get_filtered_srx_metadata(
            conn,
            organism="Homo sapiens",
            is_single_cell="yes",
            limit=100
        )
        print(metadata)