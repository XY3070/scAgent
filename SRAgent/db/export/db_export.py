import sqlite3
import psycopg2
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from psycopg2.extensions import connection

from ..get import get_prefiltered_datasets_functional
from SRAgent.db.categorization_logic import categorize_datasets_by_project

logger = logging.getLogger(__name__)


def export_prefiltered_datasets_to_postgres(
    source_conn: connection,
    export_db_name: str = "prefiltered_datasets",
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 1000,
    min_sc_confidence: int = 2,
    create_categorized_tables: bool = True
) -> Dict[str, Any]:
    """
    Export prefiltered datasets to a new PostgreSQL database
    
    Args:
        source_conn: Source database connection
        export_db_name: Name of the export database
        organisms: List of organisms to filter
        search_term: Search term for filtering
        limit: Maximum number of records
        min_sc_confidence: Minimum single-cell confidence
        create_categorized_tables: Whether to create separate tables for each category
    
    Returns:
        Dictionary containing export results and metadata
    """
    try:
        # Get prefiltered datasets
        logger.info("Getting prefiltered datasets...")
        df = get_prefiltered_datasets_functional(
            conn=source_conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit,
            min_sc_confidence=min_sc_confidence
        )
        
        if df.empty:
            logger.warning("No datasets found with current filters")
            return {"status": "no_data", "message": "No datasets found"}
        
        # Create export database connection
        try:
            # Get connection parameters from source connection
            conn_params = source_conn.get_dsn_parameters()
            export_conn_params = conn_params.copy()
            export_conn_params['database'] = export_db_name
            
            # Try to connect to export database, create if doesn't exist
            try:
                export_conn = psycopg2.connect(**export_conn_params)
            except psycopg2.OperationalError:
                # Database doesn't exist, create it
                logger.info(f"Creating database {export_db_name}...")
                admin_conn_params = conn_params.copy()
                admin_conn_params['database'] = 'postgres'  # Connect to default database
                admin_conn = psycopg2.connect(**admin_conn_params)
                admin_conn.autocommit = True
                cursor = admin_conn.cursor()
                cursor.execute(f'CREATE DATABASE "{export_db_name}"')
                cursor.close()
                admin_conn.close()
                
                # Now connect to the new database
                export_conn = psycopg2.connect(**export_conn_params)
                
        except Exception as e:
            logger.error(f"Failed to create export database: {e}")
            return {"status": "error", "message": f"Database creation failed: {e}"}
        
        try:
            # Create main table with all prefiltered data
            cursor = export_conn.cursor()
            
            # Drop and create main table
            cursor.execute("DROP TABLE IF EXISTS prefiltered_datasets")
            
            # Create table schema based on DataFrame columns
            columns_def = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    columns_def.append(f'"{col}" TEXT')
                elif df[col].dtype == 'int64':
                    columns_def.append(f'"{col}" INTEGER')
                elif df[col].dtype == 'float64':
                    columns_def.append(f'"{col}" REAL')
                else:
                    columns_def.append(f'"{col}" TEXT')
            
            create_sql = f"CREATE TABLE prefiltered_datasets ({', '.join(columns_def)})"
            cursor.execute(create_sql)
            
            # Insert data
            if not df.empty:
                placeholders = ', '.join(['%s'] * len(df.columns))
                insert_sql = f"INSERT INTO prefiltered_datasets VALUES ({placeholders})"
                data_tuples = [tuple(row) for row in df.values]
                cursor.executemany(insert_sql, data_tuples)
            
            # Create categorized tables if requested
            if create_categorized_tables:
                logger.info("Creating categorized tables...")
                categorized = categorize_datasets_by_project(df)
                
                for category, records in categorized.items():
                    if not records:
                        continue
                        
                    table_name = f"datasets_{category.lower().replace('-', '_')}"
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    cursor.execute(create_sql.replace("prefiltered_datasets", table_name))
                    
                    if records:
                        # Convert records back to DataFrame for consistent handling
                        cat_df = pd.DataFrame(records)
                        data_tuples = [tuple(row) for row in cat_df.values]
                        insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                        cursor.executemany(insert_sql, data_tuples)
            
            # Create metadata table
            cursor.execute("DROP TABLE IF EXISTS export_metadata")
            cursor.execute("""
                CREATE TABLE export_metadata (
                    export_timestamp TIMESTAMP,
                    total_records INTEGER,
                    organisms TEXT,
                    search_term TEXT,
                    record_limit INTEGER,
                    min_sc_confidence INTEGER
                )
            """)
            
            # Insert metadata
            cursor.execute("""
                INSERT INTO export_metadata VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                datetime.now(),
                len(df),
                ','.join(organisms),
                search_term,
                limit,
                min_sc_confidence
            ))
            
            export_conn.commit()
            cursor.close()
            
            logger.info(f"Successfully exported {len(df)} records to database {export_db_name}")
            
            return {
                "status": "success",
                "database_name": export_db_name,
                "total_records": len(df),
                "tables_created": ["prefiltered_datasets", "export_metadata"] + 
                               ([f"datasets_{cat.lower().replace('-', '_')}" for cat in categorized.keys() if categorized[cat]] if create_categorized_tables else [])
            }
            
        finally:
            export_conn.close()
            
    except Exception as e:
        logger.error(f"Failed to export to PostgreSQL: {e}")
        return {"status": "error", "message": str(e)}

def export_prefiltered_datasets_to_sqlite(
    conn: connection,
    output_path: str = "prefiltered_datasets.db",
    organisms: List[str] = ["human"],
    search_term: Optional[str] = None,
    limit: int = 1000,
    min_sc_confidence: int = 2,
    create_categorized_tables: bool = True
) -> Dict[str, Any]:
    """
    Export prefiltered datasets to SQLite database (lighter alternative to PostgreSQL)
    
    Args:
        conn: Source database connection
        output_path: Path to SQLite database file
        organisms: List of organisms to filter
        search_term: Search term for filtering
        limit: Maximum number of records
        min_sc_confidence: Minimum single-cell confidence
        create_categorized_tables: Whether to create separate tables for each category
    
    Returns:
        Dictionary containing export results and metadata
    """
    try:
        # Get prefiltered datasets
        logger.info("Getting prefiltered datasets...")
        df = get_prefiltered_datasets_functional(
            conn=conn,
            organisms=organisms,
            search_term=search_term,
            limit=limit,
            min_sc_confidence=min_sc_confidence
        )
        
        if df.empty:
            logger.warning("No datasets found with current filters")
            return {"status": "no_data", "message": "No datasets found"}
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to SQLite database
        sqlite_conn = sqlite3.connect(str(output_path))
        
        try:
            # Export main table
            df.to_sql('prefiltered_datasets', sqlite_conn, if_exists='replace', index=False)
            
            tables_created = ['prefiltered_datasets']
            
            # Create categorized tables if requested
            if create_categorized_tables:
                logger.info("Creating categorized tables...")
                categorized = categorize_datasets_by_project(df)
                
                for category, records in categorized.items():
                    if not records:
                        continue
                        
                    table_name = f"datasets_{category.lower().replace('-', '_')}"
                    cat_df = pd.DataFrame(records)
                    cat_df.to_sql(table_name, sqlite_conn, if_exists='replace', index=False)
                    tables_created.append(table_name)
            
            # Create metadata table
            metadata_df = pd.DataFrame([{
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'organisms': ','.join(organisms),
                'search_term': search_term,
                'record_limit': limit,
                'min_sc_confidence': min_sc_confidence
            }])
            metadata_df.to_sql('export_metadata', sqlite_conn, if_exists='replace', index=False)
            tables_created.append('export_metadata')
            
            sqlite_conn.commit()
            
            logger.info(f"Successfully exported {len(df)} records to SQLite database {output_path}")
            
            return {
                "status": "success",
                "output_path": str(output_path),
                "total_records": len(df),
                "tables_created": tables_created
            }
            
        finally:
            sqlite_conn.close()
            
    except Exception as e:
        logger.error(f"Failed to export to SQLite: {e}")
        return {"status": "error", "message": str(e)}
