from SRAgent.db.connect import db_connect
from SRAgent.db.get import db_get_table_data

if __name__ == "__main__":
    with db_connect() as conn:
        df = db_get_table_data(conn, 'srx_metadata')
        print(df.to_markdown(index=False))