from SRAgent.db.connect import db_connect
from SRAgent.db.get import db_get_table_data

# 这是一个用于检查SRX元数据（srx_metadata）的脚本。
# 它连接到数据库，获取srx_metadata表的数据，并将其打印为Markdown格式。
if __name__ == "__main__":
    # 建立数据库连接
    with db_connect() as conn:
        # 从srx_metadata表获取数据
        df = db_get_table_data(conn, 'srx_metadata')
        # 将数据框打印为Markdown格式，不包含索引
        print(df.to_markdown(index=False))