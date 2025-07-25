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
    limit: int = 100  # 预筛选上限
) -> list:
    """
    Performs a pre-filtering query against the local sra_geo_ft table.
    Note: Date filtering is now based on "gsm_submission_date" instead of "submission_date",
      to avoid invalid date values ('0000-00-00') in the original field.
    """
    query_parts = [
        """
        SELECT "sra_ID", study_title, summary, overall_design, scientific_name, library_strategy
        FROM merged.sra_geo_ft
        WHERE 1=1
        """
    ]

    params = []
    # 添加条件时，每个条件前加 "AND"
    conditions = []

    # 1. 物种筛选 (Organism Filtering)
    if "human" in organisms:
        conditions.append("""
            (organism_ch1 ILIKE %s OR scientific_name ILIKE %s OR organism ILIKE %s OR source_name_ch1 ILIKE %s OR common_name ILIKE %s)
        """)
        params.extend(['%homo sapiens%', '%homo sapiens%', '%homo sapiens%', '%human%', '%human%'])

    if "mouse" in organisms:
        conditions.append("""
            (scientific_name = %s OR "organism_ch1" = %s OR common_name = %s)
        """)
        params.extend(['Mus musculus', 'mouse', 'mouse'])

    # 2. 单细胞筛选 (Single-Cell Filtering)
    sc_keywords = [
        "%scRNA%", "%single cell%", "%10x%", "%10X%", "%Chromium%",
        "%C1 Fluidigm%", "%Smart-seq%", "%microwell%", "%droplet%",
        "%inDrop%", "%Seq-Well%", "%Fluidigm%"
    ]
    sc_placeholders = ", ".join(["%s"] * len(sc_keywords))
    conditions.append(f"""
        (
            library_strategy ILIKE ANY(ARRAY[{sc_placeholders}]) OR
            technology ILIKE ANY(ARRAY[{sc_placeholders}]) OR
            characteristics_ch1 ILIKE ANY(ARRAY[{sc_placeholders}]) OR
            summary ILIKE ANY(ARRAY[{sc_placeholders}]) OR
            overall_design ILIKE ANY(ARRAY[{sc_placeholders}])
        )
    """)
    params.extend(sc_keywords * 5)  # 添加五次，因为有五个 ILIKE ANY

    # 3. 数据可用性筛选 (Data Availability Filtering)
    conditions.append('("sra_ID" IS NOT NULL OR study_xref_link IS NOT NULL)')

    # 4. 日期筛选 (Date Filtering)
    if min_date and min_date.strip():
        conditions.append("gsm_submission_date::date >= %s")
        params.append(min_date)
    if max_date and max_date.strip():
        conditions.append("gsm_submission_date::date <= %s")
        params.append(max_date)

    # 5. 语义关键词筛选 (Semantic Keyword Filtering)
    if search_term and search_term.strip():
        search_fields = ["study_title", "summary", "gse_title", "gsm_title", "overall_design", '"characteristics_ch1"']
        combined_search_fields = " || ' ' || ".join(search_fields)
        conditions.append(f"({combined_search_fields}) ILIKE %s")
        params.append(f"%{search_term}%")
    
    # 拼接查询
    if conditions:
        query_parts.extend([f"AND {cond}" for cond in conditions])
    query_parts.append(f"LIMIT {limit}")
    # 拼接最终 SQL
    final_query = " ".join(query_parts)

    # 执行查询
    try:
        df = pd.read_sql(final_query, conn, params=params)
        print(final_query)
        print(f"Found {len(df)} prefiltered datasets.", file=sys.stderr)
        return df.to_dict(orient='records') if not df.empty else []
    except Exception as e:
        print(f"Error executing pre-filtering query: {e}", file=sys.stderr)
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
