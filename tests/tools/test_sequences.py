import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from SRAgent.tools.sequences import fastq_dump, sra_stat

@pytest.fixture
def mock_fastq_content():
    """Sample fastq content for tests"""
    return """@SRR31573627.1 1 length=151
GTACGTAGCTAGCTAGCTAGCTACGATCGATGCATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTGACTGATCGTAGCTAGCTACGATCGTAGCTAGCTAGCTACGTACGTAGCTAGCTACGATCGTAGCTAGCT
+
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
@SRR31573627.2 2 length=151
ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT
+
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
"""

@pytest.fixture
def mock_sra_stat_xml():
    """Sample XML output from sra-stat"""
    return """<?xml version="1.0" encoding="UTF-8" ?>
<stats>
  <stat name="bioproject" value="PRJNA123456" />
  <stat name="biosample" value="SAMN12345678" />
  <stat name="organism" value="Homo sapiens" />
  <run accession="SRR31573627" spots="1000000" bases="151000000" avgLength="151">
    <statistics>
      <readCount>1000000</readCount>
      <baseCount>151000000</baseCount>
      <layout>PAIRED</layout>
    </statistics>
  </run>
</stats>
"""

def test_fastq_dump_successful(mock_fastq_content, tmp_path):
    """Test fastq_dump with successful execution"""
    # Create mock files in temp dir
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    file1_path = test_dir / "SRR31573627_1.fastq"
    file2_path = test_dir / "SRR31573627_2.fastq"
    
    file1_path.write_text(mock_fastq_content)
    file2_path.write_text(mock_fastq_content)
    
    mock_temp = MagicMock()
    type(mock_temp.return_value).name = PropertyMock(return_value=str(test_dir))
    
    with patch('SRAgent.tools.sequences.tempfile.TemporaryDirectory', mock_temp), \
         patch('SRAgent.tools.sequences.which', return_value=True), \
         patch('SRAgent.tools.sequences.run_cmd') as mock_run:
        
        mock_run.return_value = (0, b"", b"")
        
        # Call function
        result = fastq_dump.invoke({"SRR_accessions": ["SRR31573627"]})
        
        # Verify fastq-dump was called with correct args
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "fastq-dump"
        assert "--split-files" in args
        assert "SRR31573627" in args
        
        # Verify content of result
        assert "#-- File: SRR31573627_1.fastq --#" in result
        assert "#-- File: SRR31573627_2.fastq --#" in result
        assert "@SRR31573627.1" in result
        assert "@SRR31573627.2" in result

def test_fastq_dump_multiple_accessions(tmp_path):
    """Test fastq_dump with multiple accessions"""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    
    mock_temp = MagicMock()
    type(mock_temp.return_value).name = PropertyMock(return_value=str(test_dir))
    
    with patch('SRAgent.tools.sequences.tempfile.TemporaryDirectory', mock_temp), \
         patch('SRAgent.tools.sequences.which', return_value=True), \
         patch('SRAgent.tools.sequences.run_cmd') as mock_run:
        
        mock_run.return_value = (0, b"", b"")
        
        # Create test files
        for acc in ["SRR31573627", "SRR31573628"]:
            (test_dir / f"{acc}_1.fastq").touch()
        
        # Call function
        result = fastq_dump.invoke({"SRR_accessions": ["SRR31573627", "SRR31573628"]})
        
        # Verify fastq-dump was called with both accessions
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "SRR31573627" in args
        assert "SRR31573628" in args
        
        # Verify files were processed
        assert "SRR31573627_1.fastq" in result
        assert "SRR31573628_1.fastq" in result

def test_fastq_dump_tool_not_installed():
    """Test fastq_dump when the tool is not installed"""
    with patch('SRAgent.tools.sequences.which', return_value=False):
        result = fastq_dump.invoke({"SRR_accessions": ["SRR31573627"]})
        assert "fastq-dump is not installed" in result

def test_fastq_dump_invalid_accession():
    """Test fastq_dump with invalid accession"""
    with patch('SRAgent.tools.sequences.which', return_value=True):
        result = fastq_dump.invoke({"SRR_accessions": ["ERR12345"]})
        assert "Invalid SRA accession" in result

def test_fastq_dump_command_error():
    """Test fastq_dump when command fails"""
    mock_temp = MagicMock()
    type(mock_temp.return_value).name = PropertyMock(return_value="/tmp/test")
    
    with patch('SRAgent.tools.sequences.tempfile.TemporaryDirectory', mock_temp), \
         patch('SRAgent.tools.sequences.which', return_value=True), \
         patch('SRAgent.tools.sequences.run_cmd') as mock_run:
        
        mock_run.return_value = (1, b"", b"Error: accession not found")
        
        result = fastq_dump.invoke({"SRR_accessions": ["SRR31573627"]})
        
        assert "Error running fastq-dump" in result

def test_sra_stat_successful(mock_sra_stat_xml):
    """Test sra_stat with successful execution"""
    with patch('SRAgent.tools.sequences.which', return_value=True), \
         patch('SRAgent.tools.sequences.run_cmd') as mock_run:
        
        mock_run.return_value = (0, mock_sra_stat_xml.encode(), b"")
        
        result = sra_stat.invoke({"accessions": ["SRR31573627"]})
        
        # Verify sra-stat was called with correct args
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "sra-stat"
        assert "--xml" in args
        assert "SRR31573627" in args
        
        # Verify content of result
        assert "PRJNA123456" in result
        assert "Homo sapiens" in result
        assert "PAIRED" in result
        assert "SRR31573627" in result

def test_sra_stat_multiple_accessions():
    """Test sra_stat with multiple accessions"""
    with patch('SRAgent.tools.sequences.which', return_value=True), \
         patch('SRAgent.tools.sequences.run_cmd') as mock_run:
        
        mock_run.return_value = (0, b"<stats><run accession='SRR31573627'></run><run accession='SRP309720'></run></stats>", b"")
        
        result = sra_stat.invoke({"accessions": ["SRR31573627", "SRP309720"]})
        
        # Verify sra-stat was called with both accessions
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "SRR31573627" in args
        assert "SRP309720" in args
        assert isinstance(result, str)

def test_sra_stat_tool_not_installed():
    """Test sra_stat when the tool is not installed"""
    with patch('SRAgent.tools.sequences.which', return_value=False):
        result = sra_stat.invoke({"accessions": ["SRR31573627"]})
        assert "sra-stat is not installed" in result

def test_sra_stat_invalid_accession():
    """Test sra_stat with invalid accession"""
    with patch('SRAgent.tools.sequences.which', return_value=True):
        result = sra_stat.invoke({"accessions": ["XXX12345"]})
        assert "Invalid GEO/SRA accession" in result

def test_sra_stat_command_error():
    """Test sra_stat when command fails"""
    with patch('SRAgent.tools.sequences.which', return_value=True), \
         patch('SRAgent.tools.sequences.run_cmd') as mock_run:
        
        mock_run.return_value = (1, b"", b"Error: accession not found")
        result = sra_stat.invoke({"accessions": ["SRR31573627"]})
        
        assert "Error running sra-stat" in result