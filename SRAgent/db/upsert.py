# import
## batteries
import os
from typing import List, Dict, Any, Tuple, Optional
## 3rd party
import pandas as pd
from pypika import Query, Table, Field, Column, Criterion
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import connection
## package
from SRAgent.db.utils import get_unique_columns

# functions
def db_upsert(df: pd.DataFrame, table_name: str, conn: connection) -> None:
    """
    Upload a pandas DataFrame to PostgreSQL, performing an upsert operation.
    If records exist (based on unique constraints), update them; otherwise insert new records.
    Args:
        df: pandas DataFrame to upload
        table_name: name of the target table
        conn: psycopg2 connection object
    """   
    # if df is empty, return
    if df.empty:
        return
    # if df is not dataframe, try to convert
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            raise Exception(f"Error converting input to DataFrame: {str(e)}")

    # Get DataFrame columns
    columns = list(df.columns)
    
    # Create ON CONFLICT clause based on unique constraints
    unique_columns = get_unique_columns(table_name, conn)

    # Exclude 'id' column from the upsert
    if "id" in columns:
        df = df.drop(columns=["id"])
        columns.remove("id")

    # Drop duplicate records based on unique columns
    df.drop_duplicates(subset=unique_columns, keep='first', inplace=True)

    # Convert DataFrame to list of tuples
    values = [tuple(x) for x in df.to_numpy()]

    # Create the INSERT statement with ON CONFLICT clause
    insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)})"
    insert_stmt += f"\nVALUES %s"

    # Add DO UPDATE SET clause for non-unique columns
    do_update_set = [col for col in columns if col not in unique_columns]
    if do_update_set:
        do_update_set = ', '.join(f"{col} = EXCLUDED.{col}" for col in do_update_set)
        insert_stmt += f"\nON CONFLICT ({', '.join(unique_columns)})"
        insert_stmt += f"\nDO UPDATE SET {do_update_set}"
    else:
        # if no non-unique columns, add DO NOTHING clause
        insert_stmt += f"\nON CONFLICT ({', '.join(unique_columns)}) DO NOTHING"

    # Execute the query
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_stmt, values)
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error uploading data to {table_name}:\n{str(e)}\n\nSQL:{insert_stmt}\n\nValues:{str(values)}")
        raise Exception(f"Error uploading data to {table_name}:\n{str(e)}\n\nSQL:{insert_stmt}\n\nValues:{str(values)}")

# main
if __name__ == '__main__':
    from dotenv import load_dotenv
    from SRAgent.db.connect import db_connect
    load_dotenv()

    # test data
    df = pd.DataFrame({
        "database": ["sra", "sra"],
        "entrez_id": [25200088, 25200089],
        "srx_accession": ["SRX18216984", "SRX18216985"],
        "is_single_cell": ["yes", "yes"],
        "organism": ["Homo sapiens", "Homo sapiens"],
        "tech_10x": ["10x_genomics", "10x_genomics"],
        "lib_prep": ["Smart-seq2", "Smart-seq2"],
        "tissue": ["blood", "brain"],
        "disease": ["none", "Alzheimer's Disease"],
        "cell_line": ["none", "none"],
        "library_strategy": ["RNA-Seq", "RNA-Seq"],
        "library_source": ["TRANSCRIPTOMIC", "TRANSCRIPTOMIC"],
        "library_selection": ["cDNA", "cDNA"],
        "platform": ["ILLUMINA", "ILLUMINA"],
        "instrument_model": ["Illumina NovaSeq 6000", "Illumina NovaSeq 6000"],
        "sra_study_accession": ["SRP000001", "SRP000002"],
        "bioproject_accession": ["PRJNA000001", "PRJNA000002"],
        "biosample_accession": ["SAMN000001", "SAMN000002"],
        "pubmed_id": ["12345678", "87654321"],
        "title": ["Human single cell RNA-seq of blood", "Single cell RNA-seq of human brain with AD"],
        "design_description": ["Single cell RNA-seq", "Single cell RNA-seq"],
        "sample_description": ["Healthy human blood sample", "Human brain tissue from AD patient"],
        "submission_date": ["2023-01-01", "2023-02-01"],
        "update_date": ["2023-01-01", "2023-02-01"],
        "czi_collection_id": ["czi_id_1", "czi_id_2"],
        "czi_collection_name": ["Human Blood Atlas", "Human Brain Atlas"],
        "notes": ["Test data 1", "Test data 2"]
    })
    with db_connect() as conn:
        db_upsert(df, "srx_metadata", conn)