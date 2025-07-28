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

# def get_db_certs(certs=["server-ca.pem", "client-cert.pem", "client-key.pem"]) -> dict:
#     """
#     Download certificates from GCP Secret Manager and save them to temporary files.
#     Args:
#         certs: A list of certificate ids
#     Returns:
#         A dictionary containing the paths to the temporary files
#     """
#     idx = {
#         "server-ca.pem": "SRAgent_db_server_ca",
#         "client-cert.pem": "SRAgent_db_client_cert",
#         "client-key.pem": "SRAgent_db_client_key"
#     }
#     cert_files = {}
#     for cert in certs:
#         cert_files[cert] = download_secret(idx[cert])
#     return cert_files

# def download_secret(secret_id: str) -> str:
#     """
#     Download a secret from GCP Secret Manager and save it to a temporary file.
#     Args:
#         secret_id: The secret id
#     Returns:
#         The path to the temporary file containing the secret
#     """
#     secret_value = get_secret(secret_id)
#     temp_file = NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
#     with temp_file as f:
#         f.write(secret_value)
#         f.flush()
#     return temp_file.name

# def get_secret(secret_id: str) -> str:
#     """
#     Fetch secret from GCP Secret Manager. Falls back to environment variable if secret cannot be obtained.
#     Required environment variables: GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS
#     Args:
#         secret_id: The secret id
#     Returns:
#         The secret value
#     """
#     try:
#         from google.auth import default
#         from google.cloud import secretmanager

#         _, project_id = default()  # Use default credentials; project_id is inferred
#         if not project_id:
#             project_id = os.environ["GCP_PROJECT_ID"]
#         name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
#         client = secretmanager.SecretManagerServiceClient()
#         response = client.access_secret_version(request={"name": name})
#         return response.payload.data.decode('UTF-8')
#     except Exception as e:
#         # Fall back to environment variable
#         env_var = os.getenv(secret_id)
#         if env_var is not None:
#             return env_var
#         raise Exception(f"Failed to get secret '{secret_id}' from Secret Manager and environment variable not set") from e


# main
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(override=True)

    with db_connect() as conn:
       print(conn)
       print("Database connection established!")
    