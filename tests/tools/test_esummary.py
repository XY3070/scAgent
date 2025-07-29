import pytest
from unittest.mock import Mock, patch
from Bio import Entrez
from Bio.Entrez.Parser import ValidationError
from SRAgent.tools.esummary import esummary


@pytest.fixture
def mock_esummary():
    """Fixture to mock Entrez.esummary and time.sleep."""
    with patch('Bio.Entrez.esummary') as mock_es, patch('time.sleep') as mock_sleep:
        yield mock_es, mock_sleep

def test_successful_esummary_single_id(mock_esummary):
    """Test esummary with a single valid Entrez ID."""
    mock_es, mock_sleep = mock_esummary
    
    # Mock response
    mock_handle = Mock()
    mock_handle.read.return_value = b'<eSummaryResult><DocSum><Id>29110018</Id></DocSum></eSummaryResult>'
    mock_es.return_value = mock_handle
    
    # Call function
    result = esummary.invoke({"entrez_ids": ["29110018"], "database": "sra"})
    
    # Verify results
    assert "DocSum" in result
    assert "29110018" in result
    mock_es.assert_called_once()
    mock_sleep.assert_called_once()

def test_successful_esummary_multiple_ids(mock_esummary):
    """Test esummary with multiple valid Entrez IDs."""
    mock_es, _ = mock_esummary
    
    # Mock response
    mock_handle = Mock()
    mock_handle.read.return_value = b'<eSummaryResult><DocSum><Id>29110018</Id></DocSum><DocSum><Id>29110015</Id></DocSum></eSummaryResult>'
    mock_es.return_value = mock_handle
    
    # Call function
    result = esummary.invoke({"entrez_ids": ["29110018", "29110015"], "database": "sra"})
    
    # Verify results
    assert "29110018" in result
    assert "29110015" in result
    mock_es.assert_called_once()
def test_esummary_with_validation_error(mock_esummary):
    """Test esummary handling of ValidationError."""
    mock_es, _ = mock_esummary
    mock_es.side_effect = ValidationError
    mock_es.side_effect = Entrez.Parser.ValidationError
    
    result = esummary.invoke({"entrez_ids": ["invalid_id"], "database": "sra"})
    assert "Failed to fetch summary for IDs: invalid_id" in result
    mock_es.assert_called_once()

@pytest.mark.parametrize("database", ["sra", "gds", "pubmed"])
def test_esummary_different_databases(mock_esummary, database):
    """Test esummary with different databases."""
    mock_es, _ = mock_esummary
    
    mock_handle = Mock()
    mock_handle.read.return_value = b'<eSummaryResult><DocSum><Id>29110018</Id></DocSum></eSummaryResult>'
    mock_es.return_value = mock_handle
    
    result = esummary.invoke({"entrez_ids": ["29110018"], "database": database})
    assert "DocSum" in result
    mock_es.assert_called_with(db=database, id="29110018", retmode="xml")

def test_truncate_long_values(mock_esummary):
    """Test that long values in the response are truncated."""
    mock_es, _ = mock_esummary
    
    # Create a response with a long value
    long_value = "x" * 2000  # Longer than max_length (1000)
    mock_handle = Mock()
    mock_handle.read.return_value = f'<eSummaryResult><DocSum><Item>{long_value}</Item></DocSum></eSummaryResult>'.encode()
    mock_es.return_value = mock_handle
    
    result = esummary.invoke({"entrez_ids": ["29110018"], "database": "sra"})
    assert "...[truncated]" in result
    assert len(result) < 2000
