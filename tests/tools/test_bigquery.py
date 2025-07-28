import pytest
from unittest.mock import MagicMock
from google.cloud import bigquery
from SRAgent.tools.bigquery import (
    get_study_metadata,
    get_experiment_metadata,
    get_run_metadata,
    get_study_experiment_run
)

@pytest.fixture
def mock_client():
    """Create a mock BigQuery client"""
    client = MagicMock(spec=bigquery.Client)
    return client

@pytest.fixture
def config(mock_client):
    """Create a RunnableConfig with mock client"""
    return {"configurable": {"client": mock_client}}

def test_get_study_metadata(mock_client, config):
    """Test get_study_metadata function"""
    # Mock the query result
    mock_result = [
        {
            "sra_study": "SRP548813",
            "bioproject": "PRJNA1234567",
            "experiments": "SRX26939191,SRX26939192"
        }
    ]
    mock_client.query.return_value = mock_result

    # Invoke function
    result = get_study_metadata.invoke({
        "study_accessions": ["SRP548813"]
    }, config=config)

    # Verify query results
    assert isinstance(result, str)
    assert "SRP548813" in result
    assert "PRJNA1234567" in result
    assert "SRX26939191" in result

def test_get_experiment_metadata(mock_client, config):
    """Test get_experiment_metadata function"""
    # Mock the query result
    mock_result = [
        {
            "experiment": "SRX26939191",
            "sra_study": "SRP548813",
            "library_name": "lib1",
            "librarylayout": "paired",
            "libraryselection": "PCR",
            "librarysource": "transcriptomic",
            "platform": "ILLUMINA",
            "instrument": "NovaSeq 6000",
            "acc": "SRR31573627,SRR31573628"
        }
    ]
    mock_client.query.return_value = mock_result

    # Invoke function
    result = get_experiment_metadata.invoke({
        "experiment_accessions": ["SRX26939191"]
    }, config=config)

    # Verify query results
    assert isinstance(result, str)
    assert "SRX26939191" in result
    assert "SRP548813" in result
    assert "paired" in result
    assert "ILLUMINA" in result
    assert "SRR31573627" in result

def test_get_run_metadata(mock_client, config):
    """Test get_run_metadata function"""
    # Mock the query result
    mock_result = [
        {
            "acc": "SRR31573627",
            "experiment": "SRX26939191",
            "biosample": "SAMN12345678",
            "organism": "Homo sapiens",
            "assay_type": "RNA-Seq",
            "mbases": 1000,
            "avgspotlen": 150,
            "insertsize": 350
        }
    ]
    mock_client.query.return_value = mock_result

    # Invoke function
    result = get_run_metadata.invoke({
        "run_accessions": ["SRR31573627"]
    }, config=config)

    # Verify query results
    assert isinstance(result, str)
    assert "SRR31573627" in result
    assert "SRX26939191" in result
    assert "Homo sapiens" in result
    assert "RNA-Seq" in result

def test_get_study_experiment_run(mock_client, config):
    """Test get_study_experiment_run function"""
    # Mock the query result for different accession types
    mock_result = [
        {
            "study_accession": "SRP548813",
            "experiment_accession": "SRX26939191",
            "run_accession": "SRR31573627"
        }
    ]
    mock_client.query.return_value = mock_result

    # Test SRP accession
    result = get_study_experiment_run.invoke({
        "accessions": ["SRP548813"]
    }, config=config)
    assert isinstance(result, str)
    assert "SRP548813" in result
    assert "SRX26939191" in result
    assert "SRR31573627" in result

    # Test SRX accession
    result = get_study_experiment_run.invoke({
        "accessions": ["SRX26939191"]
    }, config=config)
    assert isinstance(result, str)
    assert "SRP548813" in result
    assert "SRX26939191" in result
    assert "SRR31573627" in result

    # Test SRR accession
    result = get_study_experiment_run.invoke({
        "accessions": ["SRR31573627"]
    }, config=config)
    assert isinstance(result, str)
    assert "SRP548813" in result
    assert "SRX26939191" in result
    assert "SRR31573627" in result

def test_study_metadata_no_results(mock_client, config):
    """Test get_study_metadata with no results"""
    mock_client.query.return_value = []
    result = get_study_metadata.invoke({
        "study_accessions": ["SRP999999"]
    }, config=config)
    assert result == "No results found"

def test_experiment_metadata_no_results(mock_client, config):
    """Test get_experiment_metadata with no results"""
    mock_client.query.return_value = []
    result = get_experiment_metadata.invoke({
        "experiment_accessions": ["SRX999999"]
    }, config=config)
    assert result == "No results found"

def test_run_metadata_no_results(mock_client, config):
    """Test get_run_metadata with no results"""
    mock_client.query.return_value = []
    result = get_run_metadata.invoke({
        "run_accessions": ["SRR999999"]
    }, config=config)
    assert result == "No results found"

def test_invalid_query_input(mock_client, config):
    """Test handling of invalid input"""
    result = get_study_experiment_run.invoke({
        "accessions": []
    }, config=config)
    assert result == "No valid accessions provided."

def test_study_experiment_run_mixed_accessions(mock_client, config):
    """Test get_study_experiment_run with mixed accession types"""
    mock_result = [
        {
            "study_accession": "SRP548813",
            "experiment_accession": "SRX26939191",
            "run_accession": "SRR31573627"
        },
        {
            "study_accession": "SRP548813",
            "experiment_accession": "SRX26939192",
            "run_accession": "SRR31573628"
        }
    ]
    mock_client.query.return_value = mock_result

    result = get_study_experiment_run.invoke({
        "accessions": ["SRP548813", "SRX26939191", "SRR31573627"]
    }, config=config)

    assert isinstance(result, str)
    assert "SRP548813" in result
    assert "SRX26939191" in result
    assert "SRX26939192" in result
    assert "SRR31573627" in result
    assert "SRR31573628" in result