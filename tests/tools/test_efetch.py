import pytest
from Bio import Entrez
from SRAgent.tools.efetch import efetch

@pytest.mark.parametrize("test_input,expected", [
    (
        {"entrez_ids": ["29110018"], "database": "sra"},
        {"success": True, "contains": ["DocSum", "29110018"]}
    ),
    (
        {"entrez_ids": ["29110018", "29110015"], "database": "sra"},
        {"success": True, "contains": ["DocSum", "29110018", "29110015"]}
    ),
])
def test_efetch_successful(monkeypatch, test_input, expected):
    """Test successful efetch queries"""
    def mock_efetch(*args, **kwargs):
        # Create mock handle
        class MockHandle:
            def read(self):
                if len(test_input["entrez_ids"]) == 1:
                    return b'<eSummaryResult><DocSum><Id>29110018</Id></DocSum></eSummaryResult>'
                else:
                    return b'<eSummaryResult><DocSum><Id>29110018</Id></DocSum><DocSum><Id>29110015</Id></DocSum></eSummaryResult>'
            def close(self):
                pass
        return MockHandle()

    monkeypatch.setattr(Entrez, "efetch", mock_efetch)
    result = efetch.invoke(test_input)
    
    assert isinstance(result, str)
    for text in expected["contains"]:
        assert text in result

def test_efetch_validation_error(monkeypatch):
    """Test handling of validation errors"""
    from Bio.Entrez.Parser import ValidationError
    def mock_efetch(*args, **kwargs):
        raise ValidationError("test_error")

    monkeypatch.setattr(Entrez, "efetch", mock_efetch)
    result = efetch.invoke({"entrez_ids": ["invalid_id"], "database": "sra"})
    
    assert "Failed to fetch record" in result

def test_efetch_general_error(monkeypatch):
    """Test handling of general errors"""
    def mock_efetch(*args, **kwargs):
        raise Exception("Network error")

    monkeypatch.setattr(Entrez, "efetch", mock_efetch)
    result = efetch.invoke({"entrez_ids": ["29110018"], "database": "sra"})
    
    assert "Failed to fetch record" in result
    assert "Network error" in result

def test_efetch_long_values_truncation(monkeypatch):
    """Test truncation of long values"""
    def mock_efetch(*args, **kwargs):
        class MockHandle:
            def read(self):
                long_text = "x" * 2000
                return f'<eSummaryResult><DocSum><Id>29110018</Id><Item>{long_text}</Item></DocSum></eSummaryResult>'.encode()
            def close(self):
                pass
        return MockHandle()

    monkeypatch.setattr(Entrez, "efetch", mock_efetch)
    result = efetch.invoke({"entrez_ids": ["29110018"], "database": "sra"})
    
    assert len(result) < 2000
    assert "...[truncated]" in result

@pytest.mark.parametrize("database", ["sra", "gds", "pubmed"])
def test_efetch_different_databases(monkeypatch, database):
    """Test efetch with different databases"""
    def mock_efetch(*args, **kwargs):
        class MockHandle:
            def read(self):
                return b'<eSummaryResult><DocSum><Id>29110018</Id></DocSum></eSummaryResult>'
            def close(self):
                pass
        return MockHandle()

    monkeypatch.setattr(Entrez, "efetch", mock_efetch)
    result = efetch.invoke({"entrez_ids": ["29110018"], "database": database})
    
    assert "DocSum" in result
    assert isinstance(result, str)

def test_efetch_large_batch(monkeypatch):
    """Test handling of large batches of IDs"""
    def mock_efetch(*args, **kwargs):
        class MockHandle:
            def read(self):
                return b'<eSummaryResult><DocSum><Id>29110018</Id></DocSum></eSummaryResult>'
            def close(self):
                pass
        return MockHandle()

    monkeypatch.setattr(Entrez, "efetch", mock_efetch)
    
    # Create list of IDs starting with the two known ones and pad with incrementing numbers
    ids = ["29110018", "29110015"] + [str(i) for i in range(3, 251)]
    result = efetch.invoke({"entrez_ids": ids, "database": "sra"})
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert "DocSum" in result
