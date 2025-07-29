# import
## batteries
import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Set
## 3rd party
import psycopg2 
import pandas as pd
from pypika import Query, Table, Field, Column, Criterion
from psycopg2.extras import execute_values
from psycopg2.extensions import connection
## package
from SRAgent.db.utils import execute_query
import logging

# functions
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

async def get_prefiltered_datasets_from_local_db(
    conn,
    organisms: list,
    min_date: str,
    max_date: str,
    search_term: str,
    limit: int = 100
) -> list:
    """
    最终工作版本 - 只使用确认存在的字段
    """
    
    # 基础查询 - 只使用我们知道存在的字段
    base_query = """
        SELECT "sra_ID", study_title, summary, overall_design, scientific_name, 
               library_strategy, technology, characteristics_ch1, gse_title, gsm_title,
               organism_ch1, source_name_ch1, common_name, gsm_submission_date
        FROM merged.sra_geo_ft
        WHERE 1=1
    """
    
    conditions = []
    params = []
    
    # 1. 物种筛选
    if organisms and "human" in organisms:
        human_condition = """
        AND (organism_ch1 ILIKE %s OR scientific_name ILIKE %s OR organism ILIKE %s 
             OR source_name_ch1 ILIKE %s OR common_name ILIKE %s)
        """
        conditions.append(human_condition)
        params.extend(['%homo sapiens%', '%homo sapiens%', '%homo sapiens%', '%human%', '%human%'])
    
    # 2. 单细胞筛选 - 优化关键词
    sc_keywords = ['%scRNA%', '%single cell%', '%10x%', '%droplet%', '%Smart-seq%']
    sc_fields = ['library_strategy', 'technology', 'characteristics_ch1', 'summary', 'overall_design']
    
    # 构建单细胞条件
    sc_conditions_list = []
    for field in sc_fields:
        for keyword in sc_keywords:
            sc_conditions_list.append(f"{field} ILIKE %s")
            params.append(keyword)
    
    if sc_conditions_list:
        sc_condition = "AND (" + " OR ".join(sc_conditions_list) + ")"
        conditions.append(sc_condition)
    
    # 3. 基本数据可用性
    conditions.append('AND "sra_ID" IS NOT NULL AND "sra_ID" != \'\' AND gse_title IS NOT NULL')
    
    # 4. 日期筛选 - 简化版本
    if min_date and min_date.strip() and min_date != '0000-00-00':
        conditions.append('AND gsm_submission_date::date >= %s')
        params.append(min_date)
    if max_date and max_date.strip() and max_date != '0000-00-00':
        conditions.append('AND gsm_submission_date::date <= %s')
        params.append(max_date)
    
    # 5. 关键词筛选 - 只使用确认存在的字段
    if search_term and search_term.strip():
        keyword_condition = """
        AND (COALESCE(study_title, '') ILIKE %s 
             OR COALESCE(summary, '') ILIKE %s 
             OR COALESCE(gse_title, '') ILIKE %s)
        """
        conditions.append(keyword_condition)
        search_pattern = f"%{search_term}%"
        params.extend([search_pattern, search_pattern, search_pattern])
    
    # 组装完整查询
    full_query = base_query + ' '.join(conditions) + f"""
        ORDER BY gsm_submission_date DESC NULLS LAST
        LIMIT {limit}
    """
    
    print(f"🔍 查询参数数量: {len(params)}, 占位符数量: {full_query.count('%s')}")
    
    try:
        df = execute_query_with_cursor(conn, full_query, tuple(params))
        
        if df.empty:
            print("查询执行成功但没有找到匹配的记录")
            return []
        
        records = df.to_dict(orient='records')
        print(f"✅ 成功找到 {len(records)} 条记录")
        return records
            
    except Exception as e:
        print(f"❌ 查询执行错误: {e}")
        import traceback
        traceback.print_exc()
        return []


# 单细胞专用函数 - 也移除不存在的字段
async def get_single_cell_datasets_from_local_db(
    conn,
    organisms: list,
    min_date: str,
    max_date: str,
    search_term: str,
    limit: int = 100
) -> list:
    """
    单细胞专用函数 - 修复版本
    """
    
    base_query = """
        SELECT "sra_ID", study_title, summary, overall_design, scientific_name, 
               library_strategy, technology, characteristics_ch1, gse_title, gsm_title,
               organism_ch1, source_name_ch1, common_name, gsm_submission_date
        FROM merged.sra_geo_ft
        WHERE 1=1
    """
    
    conditions = []
    params = []
    
    # 1. 物种筛选
    if organisms and "human" in organisms:
        conditions.append("AND (organism_ch1 ILIKE %s OR scientific_name ILIKE %s)")
        params.extend(['%homo sapiens%', '%homo sapiens%'])
    
    # 2. 严格的单细胞筛选
    strict_sc_conditions = [
        "library_strategy ILIKE %s",  # scRNA-seq
        "library_strategy ILIKE %s",  # single cell
        "technology ILIKE %s",        # 10x
        "technology ILIKE %s",        # 10X  
        "characteristics_ch1 ILIKE %s", # single cell
        "summary ILIKE %s",           # single cell
        "overall_design ILIKE %s"     # single cell
    ]
    
    sc_keywords = ['%scRNA%', '%single cell%', '%10x%', '%10X%', '%single cell%', '%single cell%', '%single cell%']
    
    conditions.append("AND (" + " OR ".join(strict_sc_conditions) + ")")
    params.extend(sc_keywords)
    
    # 3. 数据质量筛选
    conditions.append('AND "sra_ID" IS NOT NULL AND gse_title IS NOT NULL')
    
    # 4. 日期筛选
    if min_date and min_date.strip():
        conditions.append('AND gsm_submission_date::date >= %s')
        params.append(min_date)
    if max_date and max_date.strip():
        conditions.append('AND gsm_submission_date::date <= %s')
        params.append(max_date)
    
    # 5. 关键词筛选
    if search_term and search_term.strip():
        conditions.append("AND (study_title ILIKE %s OR summary ILIKE %s)")
        search_pattern = f"%{search_term}%"
        params.extend([search_pattern, search_pattern])
    
    full_query = base_query + ' '.join(conditions) + f" ORDER BY gsm_submission_date DESC LIMIT {limit}"
    
    print(f"🧬 单细胞专用查询: 参数{len(params)}个, 占位符{full_query.count('%s')}个")
    
    try:
        df = execute_query_with_cursor(conn, full_query, tuple(params))
        
        if df.empty:
            return []
        
        records = df.to_dict(orient='records')
        print(f"🧬 找到 {len(records)} 条单细胞数据")
        return records
            
    except Exception as e:
        print(f"❌ 单细胞查询错误: {e}")
        return []


# 创建一个检查表结构的辅助函数
def check_table_structure(conn):
    """
    检查表结构，确认哪些字段真正存在
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
        
        print("📊 数据库表中实际存在的字段:")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        
        return columns
        
    except Exception as e:
        print(f"❌ 检查表结构失败: {e}")
        return []


# 最简化的测试查询
async def simple_test_query(conn):
    """
    最简化的测试，确保基本功能正常
    """
    try:
        print("🧪 执行最简化测试查询...")
        
        simple_query = """
        SELECT "sra_ID", study_title, scientific_name
        FROM merged.sra_geo_ft
        WHERE "sra_ID" IS NOT NULL
        AND study_title ILIKE %s
        LIMIT 5
        """
        
        cursor = conn.cursor()
        cursor.execute(simple_query, ('%cancer%',))
        
        if cursor.description is None:
            print("❌ 查询没有返回结果描述")
            cursor.close()
            return []
        
        colnames = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        cursor.close()
        
        print(f"✅ 简化查询成功，找到 {len(results)} 条记录")
        
        if results:
            for i, row in enumerate(results):
                record = dict(zip(colnames, row))
                print(f"  {i+1}. {record['sra_ID']} - {record['study_title'][:60]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ 简化查询也失败: {e}")
        import traceback
        traceback.print_exc()
        return []


# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from SRAgent.db.connect import db_connect
    
    os.environ["DYNACONF"] = "test"
    with db_connect() as conn:
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
