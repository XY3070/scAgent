# import
## batteries
import os
import warnings

from typing import List, Dict, Any, Tuple, Optional
from tempfile import NamedTemporaryFile
## 3rd party
import pandas as pd
import psycopg2
from pypika import Query, Table, Field, Column, Criterion
from psycopg2.extras import execute_values
from psycopg2.extensions import connection
from SRAgent.config import settings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

# functions
def db_connect() -> connection:
    """
    Connect to the sql database
    """

    # connect to db
    db_params = {
        'host': settings.DB_HOST,
        'database': settings.DB_NAME,
        'user': settings.DB_USER,
        'password': settings.DB_PASSWORD,
        'port': settings.DB_PORT,
        'sslmode': 'disable',
        'connect_timeout': settings.DB_TIMEOUT
    }
    return psycopg2.connect(**db_params)

# main
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(override=True)

    with db_connect() as conn:
       print(conn)
<<<<<<< HEAD
       print("Database connection established!")
=======
       print("Connection established successfully.")
>>>>>>> feature/feat_prefilter
    