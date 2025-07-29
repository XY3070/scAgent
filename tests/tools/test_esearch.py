import pytest
from Bio import Entrez
from SRAgent.tools.esearch import esearch
from SRAgent.tools.esearch import to_sci_name
from SRAgent.tools.esearch import esearch_batch
from SRAgent.organisms import OrganismEnum

def test_esearch_successful():
    """Test successful esearch with actual API call"""
    import socket
    try:
        # Try to connect to NCBI to check internet
        socket.create_connection(("www.ncbi.nlm.nih.gov", 80), timeout=5)
        input = {"esearch_query" : "35447314", "database" : "sra"}
        ret = esearch.invoke(input)
        assert "'Count': '1'" in ret
        assert "'IdList': ['35447314']" in ret
    except (socket.timeout, socket.error):
        pytest.skip("No internet connection available")
    
def test_esearch_invalid():
    """Test successful esearch with actual API call"""
    input = {"esearch_query" : "35447314", "database" : "geo"}
    ret = esearch.invoke(input)
    assert "Error searching" in ret

def test_to_sci_name_with_all_organisms():
    """Test to_sci_name function with all values from OrganismEnum"""
    # Test each enum value by converting its name to lowercase and checking the result
    for organism in OrganismEnum:
        # Get the enum name (e.g., 'HUMAN') and convert to lowercase
        organism_name = organism.name.lower()
        # Call to_sci_name with the lowercase name
        result = to_sci_name(organism_name)
        # Verify the result is the quoted scientific name from the enum
        expected = f'"{organism.value}"'
        assert result == expected, f"Failed for {organism_name}: expected {expected}, got {result}"
    
    # Test with spaces in names (replacing underscores with spaces)
    for organism in OrganismEnum:
        if "_" in organism.name:
            # Convert enum name to lowercase and replace underscores with spaces
            organism_name = organism.name.lower().replace("_", " ")
            # Call to_sci_name with the spaced name
            result = to_sci_name(organism_name)
            # Verify the result
            expected = f'"{organism.value}"'
            assert result == expected, f"Failed for '{organism_name}': expected {expected}, got {result}"

def test_to_sci_name_unknown_organism():
    """Test to_sci_name function with an unknown organism name"""
    with pytest.raises(ValueError, match="Organism 'nonexistent' not found in OrganismEnum"):
        to_sci_name("nonexistent")

def test_esearch_batch_large_result():
    """Test that esearch_batch can handle requests for more than 10000 records"""
    # Use a query that returns many results
    query = '("single cell RNA sequencing" OR "single cell RNA-seq")'
    # Request 15000 records
    ids = esearch_batch(
        esearch_query=query,
        database="sra",
        max_ids=15000,
        verbose=False
    )
    # Verify we got more than 10000 records
    assert len(ids) > 10000, f"Expected more than 10000 records, got {len(ids)}"
    # Verify we got exactly 15000 records
    assert len(ids) == 15000, f"Expected 15000 records, got {len(ids)}"
    # Verify all IDs are unique
    assert len(set(ids)) == len(ids), "Found duplicate IDs in results"
