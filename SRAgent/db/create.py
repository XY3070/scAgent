# import
## batteries
import os
import warnings
from typing import List, Dict, Any, Tuple, Optional
## 3rd party
import psycopg2
import pandas as pd
from pypika import Query, Table, Field, Column, Criterion
from psycopg2.extras import execute_values
from psycopg2.extensions import connection
## package
from SRAgent.db.utils import execute_query

# functions
def create_updated_at_trigger(tbl_name: str, conn: connection) -> None:

    # Define the raw SQL for the trigger function and trigger
    trigger_function_sql = """
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
    """

    trigger_sql = f"""
DROP TRIGGER IF EXISTS set_updated_at ON {tbl_name};
CREATE TRIGGER set_updated_at
BEFORE UPDATE ON {tbl_name}
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();
    
    """

    # Execute the SQL statements
    with conn.cursor() as cur:
        cur.execute(trigger_function_sql)
        cur.execute(trigger_sql)

def create_srx_metadata(conn: connection) -> None:
    tbl_name = "srx_metadata"
    # Drop table if it exists
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {tbl_name} CASCADE;")
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("database", "VARCHAR(20)", nullable=False),
            Column("entrez_id", "INT", nullable=False),
            Column("srx_accession", "VARCHAR(20)"),
            Column("is_illumina", "VARCHAR(10)"),
            Column("is_single_cell", "VARCHAR(10)"),
            Column("is_paired_end", "VARCHAR(10)"),
            Column("lib_prep", "VARCHAR(30)"),
            Column("tech_10x", "VARCHAR(30)"),
            Column("cell_prep", "VARCHAR(30)"),
            Column("organism", "VARCHAR(100)"),
            Column("tissue", "VARCHAR(300)"),
            Column("tissue_ontology_term_id", "VARCHAR(300)"),
            Column("disease", "VARCHAR(300)"),
            Column("perturbation", "VARCHAR(300)"),
            Column("cell_line", "VARCHAR(300)"),
            Column("library_strategy", "VARCHAR(50)"),
            Column("library_source", "VARCHAR(50)"),
            Column("library_selection", "VARCHAR(50)"),
            Column("platform", "VARCHAR(50)"),
            Column("instrument_model", "VARCHAR(100)"),
            Column("sra_study_accession", "VARCHAR(20)"),
            Column("bioproject_accession", "VARCHAR(20)"),
            Column("biosample_accession", "VARCHAR(20)"),
            Column("pubmed_id", "VARCHAR(20)"),
            Column("title", "TEXT"),
            Column("design_description", "TEXT"),
            Column("sample_description", "TEXT"),
            Column("submission_date", "TIMESTAMP"),
            Column("update_date", "TIMESTAMP"),
            Column("czi_collection_id", "VARCHAR(40)"),
            Column("czi_collection_name", "VARCHAR(300)"),
            Column("notes", "TEXT"),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("database", "entrez_id")
    
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_srx_srr(conn: connection) -> None:
    tbl_name = "srx_srr"
    # Drop table if it exists
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {tbl_name} CASCADE;")
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("srx_accession", "VARCHAR(20)", nullable=False),
            Column("srr_accession", "VARCHAR(20)", nullable=False),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("srx_accession", "srr_accession")
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_eval(conn: connection) -> None:
    tbl_name = "eval"
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("dataset_id", "VARCHAR(30)", nullable=False),
            Column("database", "VARCHAR(20)", nullable=False),
            Column("entrez_id", "INT", nullable=False),
            Column("srx_accession", "VARCHAR(20)"),
            Column("is_illumina", "VARCHAR(10)"),
            Column("is_single_cell", "VARCHAR(10)"),
            Column("is_paired_end", "VARCHAR(10)"),
            Column("lib_prep", "VARCHAR(30)"),
            Column("tech_10x", "VARCHAR(30)"),
            Column("cell_prep", "VARCHAR(30)"),
            Column("organism", "VARCHAR(80)"),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("dataset_id", "database", "entrez_id")
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_screcounter_log(conn: connection) -> None:
    tbl_name = "screcounter_log"
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("sample", "VARCHAR(20)", nullable=False),
            Column("accession", "VARCHAR(20)"),
            Column("process", "VARCHAR(40)", nullable=False),
            Column("step", "VARCHAR(40)", nullable=False),
            Column("status", "VARCHAR(20)", nullable=False),
            Column("message", "VARCHAR(200)", nullable=False),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("sample", "accession", "process", "step")
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_screcounter_star_params(conn: connection) -> None:
    tbl_name = "screcounter_star_params"
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("sample", "VARCHAR(20)", nullable=False),
            Column("barcodes", "VARCHAR(100)"),
            Column("star_index", "VARCHAR(100)"),
            Column("cell_barcode_length", "INT"),
            Column("umi_length", "INT"),
            Column("strand", "VARCHAR(20)"),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("sample")
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_screcounter_star_results(conn: connection) -> None:
    tbl_name = "screcounter_star_results"
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("sample", "VARCHAR(20)", nullable=False),   # aka: srx_accession
            Column("feature", "VARCHAR(30)", nullable=False),  # STAR feature type
            Column("estimated_number_of_cells", "INT"),
            Column("fraction_of_unique_reads_in_cells", "FLOAT"),
            Column("mean_gene_per_cell", "FLOAT"),
            Column("mean_umi_per_cell", "FLOAT"),
            Column("mean_feature_per_cell", "FLOAT"),
            Column("median_gene_per_cell", "FLOAT"),
            Column("median_umi_per_cell", "FLOAT"),
            Column("median_feature_per_cell", "FLOAT"),
            Column("number_of_reads", "INT"),
            Column("median_reads_per_cell", "FLOAT"),
            Column("q30_bases_in_cb_umi", "FLOAT"),
            Column("q30_bases_in_rna_read", "FLOAT"),
            Column("reads_mapped_to_gene__unique_gene", "FLOAT"),
            Column("reads_mapped_to_gene__unique_multiple_gene", "FLOAT"),
            Column("reads_mapped_to_genefull__unique_genefull", "FLOAT"),
            Column("reads_mapped_to_genefull__unique_multiple_genefull", "FLOAT"),
            Column("reads_mapped_to_genefull_ex50pas__unique_genefull_ex50pas", "FLOAT"),
            Column("reads_mapped_to_genefull_ex50pas__unique_multiple_genefull_ex50pas", "FLOAT"),
            Column("reads_mapped_to_genefull_exonoverintron__unique_genefull_exonoverintron", "FLOAT"),
            Column("reads_mapped_to_genefull_exonoverintron__unique_multiple_genefull_exonoverintron", "FLOAT"),
            Column("reads_mapped_to_genome__unique", "FLOAT"),
            Column("reads_mapped_to_genome__unique_multiple", "FLOAT"),
            Column("reads_mapped_to_velocyto__unique_velocyto", "FLOAT"),
            Column("reads_mapped_to_velocyto__unique_multiple_velocyto", "FLOAT"),
            Column("reads_with_valid_barcodes", "FLOAT"),
            Column("sequencing_saturation", "FLOAT"),
            Column("total_feature_detected", "FLOAT"),
            Column("umis_in_cells", "INT"),
            Column("unique_reads_in_cells_mapped_to_gene", "FLOAT"),
            Column("unique_reads_in_cells_mapped_to_genefull", "FLOAT"),
            Column("unique_reads_in_cells_mapped_to_genefull_ex50pas", "FLOAT"),
            Column("unique_reads_in_cells_mapped_to_genefull_exonoverintron", "FLOAT"),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("sample", "feature")
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_screcounter_trace(conn: connection) -> None:
    tbl_name = "screcounter_trace"
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("task_id", "VARCHAR(20)", nullable=False),
            Column("hash", "VARCHAR(12)", nullable=False),
            Column("native_id", "VARCHAR(80)", nullable=False),
            Column("name", "VARCHAR(255)"),
            Column("queue", "VARCHAR(40)"),
            Column("status", "VARCHAR(40)"),
            Column("exit", "VARCHAR(10)"),
            Column("submit", "VARCHAR(40)"),
            Column("container", "VARCHAR(255)"),
            Column("cpus", "INT"),
            Column("memory", "VARCHAR(20)"),
            Column("disk", "VARCHAR(20)"),
            Column("time", "VARCHAR(20)"),
            Column("duration", "VARCHAR(20)"),
            Column("attempts", "INT"),
            Column("script", "TEXT"),
            Column("error_action", "TEXT"),
            Column("error_stack", "TEXT"),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("task_id")
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_scbasecamp_metadata(conn: connection) -> None:
    tbl_name = "scbasecamp_metadata"
    stmt = Query \
        .create_table(tbl_name) \
        .columns(
            Column("entrez_id", "INT", nullable=False),
            Column("srx_accession", "VARCHAR(20)", nullable=False),
            Column("feature_type", "VARCHAR(30)", nullable=False),
            Column("file_path", "VARCHAR(200)"),
            Column("obs_count", "INT"),
            Column("lib_prep", "VARCHAR(30)"),
            Column("tech_10x", "VARCHAR(30)"),
            Column("cell_prep", "VARCHAR(30)"),
            Column("organism", "VARCHAR(100)"),
            Column("tissue", "VARCHAR(300)"),
            Column("tissue_ontology_term_id", "VARCHAR(300)"),
            Column("disease", "VARCHAR(300)"),
            Column("perturbation", "VARCHAR(300)"),
            Column("cell_line", "VARCHAR(300)"),
            Column("czi_collection_id", "VARCHAR(40)"),
            Column("czi_collection_name", "VARCHAR(300)"),
            Column("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
            Column("updated_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ) \
        .unique("entrez_id", "srx_accession", "feature_type")
    
    execute_query(stmt, conn)
    create_updated_at_trigger(tbl_name, conn)

def create_table_router() -> Dict[str, Any]:
    router = {
        "srx_metadata": create_srx_metadata,
        "srx_srr": create_srx_srr,
        "eval": create_eval,
        "screcounter_log": create_screcounter_log,
        "screcounter_star_params": create_screcounter_star_params,
        "screcounter_star_results": create_screcounter_star_results,
        "screcounter_trace": create_screcounter_trace,
        "scbasecamp_metadata": create_scbasecamp_metadata,
    }
    return router

def create_table(table_name: str, conn: connection) -> None:
    # router
    router = create_table_router()
    if table_name == "ALL":
        for table_name in router.keys():
            router[table_name](conn)
        return None

    # create the table
    if table_name in router:
        router[table_name](conn)
    else:
        raise ValueError(f"Table {table_name} not recognized")

# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    from SRAgent.db.connect import db_connect
    load_dotenv(override=True)

    # connect to db
    os.environ["DYNACONF"] = "test"
    with db_connect() as conn:
        # create tables
        #create_srx_metadata()
        create_scbasecamp_metadata(conn)