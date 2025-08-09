import pytest
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import pandas as pd
from SRAgent.cli.metadata import metadata_agent_main
from SRAgent.agents.utils import load_settings


class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@pytest.fixture
def mock_db_connect():
    with patch('SRAgent.db.connect.db_connect') as mock_conn:
        mock_conn.return_value.__enter__.return_value = MagicMock(commit=MagicMock())
        yield mock_conn

@pytest.fixture
def mock_db_get_filtered_srx_metadata():
    with patch('SRAgent.db.get.db_get_filtered_srx_metadata') as mock_get:
        yield mock_get
        
@pytest.fixture
def mock_db_add_srx_metadata():
    with patch('SRAgent.db.add.db_add_srx_metadata') as mock_add:
        yield mock_add
        
@pytest.fixture
def mock_db_add_srr_accessions():
    with patch('SRAgent.db.add.db_add_srr_accessions') as mock_add_srr:
        yield mock_add_srr

@pytest.fixture
def mock_load_settings():
    with patch('SRAgent.agents.utils.load_settings') as mock_load:
        mock_load.return_value = {
            'prod': {
                'db_host': 'mock_db_host',
                'db_name': 'mock_db_name',
                'db_user': 'mock_db_user',
                'db_password': 'mock_db_password',
                'db_port': 5432,
                'db_timeout': 60
            },
            'models': {
                'sragent': 'mock_model_name'
            }
        }
        yield mock_load

def test_sql_filter_functionality(mock_db_connect, mock_db_get_filtered_srx_metadata, mock_db_add_srx_metadata, mock_db_add_srr_accessions, mock_load_settings):
    # Mock the return value of db_get_filtered_srx_metadata
    mock_db_get_filtered_srx_metadata.return_value = pd.DataFrame([
        {
            "entrez_id": 123,
            "srx_accession": "SRX123",
            "organism": "Homo sapiens",
            "is_single_cell": "yes",
            "title": "Single cell RNA-seq of human lung",
            "design_description": "10x Genomics single cell sequencing"
        },
        {
            "entrez_id": 456,
            "srx_accession": "SRX456",
            "organism": "Mus musculus",
            "is_single_cell": "no",
            "title": "Bulk RNA-seq of mouse brain",
            "design_description": "Illumina RNA sequencing"
        }
    ])

    # Simulate command line arguments for filtering
    args = MockArgs(
        from_db=True,
        srx_accession_csv=None,
        database="sra",
        max_concurrency=6,
        recursion_limit=200,
        max_parallel=2,
        no_srr=False,
        use_database=False,
        write_graph=None,
        output_csv=None,
        output_file=None,
        limit=10,
        filter_by=[],
        no_summaries=True,
        query="human lung",
        organism="Homo sapiens",
        single_cell="true",
        keywords=None
    )

    # Run the main function
    metadata_agent_main(args)

    # Assert that db_get_filtered_srx_metadata was called with the correct arguments
    mock_db_get_filtered_srx_metadata.assert_called_once_with(
        conn=mock_db_connect.return_value.__enter__.return_value,
        organism="Homo sapiens",
        is_single_cell="true",
        keywords=None,
        query="human lung",
        limit=10,
        database="sra"
    )

    # You can add more assertions here to check the output or other side effects
    # For example, if you were writing to a file, you could check file contents.
    # Since this test focuses on the database call, we primarily check the mock call.

if __name__ == '__main__':
    # This allows running the test directly as a script
    # It's a simplified way to run the test without pytest runner
    # For full pytest features, use pytest command
    # Mock db_connect
    # Mock db_get_filtered_srx_metadata
    with patch('SRAgent.db.get.db_get_filtered_srx_metadata') as mock_db_get_filtered_srx_metadata:
        mock_db_get_filtered_srx_metadata.return_value = pd.DataFrame([
            {'srx_id': 'SRX123', 'organism': 'Homo sapiens', 'single_cell': True, 'keywords': 'test'}
        ])
        # Mock load_settings
        with patch('SRAgent.agents.utils.load_settings') as mock_load_settings:
            mock_load_settings.return_value = {
                'prod': {
                    'db_host': 'mock_db_host',
                    'db_name': 'mock_db_name',
                    'db_user': 'mock_db_user',
                    'db_password': 'mock_db_password',
                    'db_port': 5432,
                    'db_timeout': 60
                },
                'models': {
                    'sragent': 'Qwen3-235B-A22B',
                    'entrez': 'Qwen3-235B-A22B',
                    'esearch': 'Qwen3-235B-A22B',
                    'esummary': 'Qwen3-235B-A22B',
                    'default': 'Qwen3-235B-A22B'
                },
                'temperature': {
                    'sragent': 0.1,
                    'entrez': 0.1,
                    'esearch': 0.1,
                    'esummary': 0.1,
                    'default': 0.1
                },
                'qwen_api_base': 'http://mock_qwen_api_base',
                'qwen_api_key': 'mock_qwen_api_key'
            }
            # Pass a dummy mock_db_connect as it's no longer used directly in this block
            test_sql_filter_functionality(MagicMock(), mock_db_get_filtered_srx_metadata, MagicMock(), MagicMock(), mock_load_settings)