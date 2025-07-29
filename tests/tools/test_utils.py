import os
import xml.etree.ElementTree as ET
import pytest
from SRAgent.tools.utils import batch_ids, truncate_values, xml2json, run_cmd, to_json, join_accs

def test_batch_ids():
    """Test batch_ids function"""
    # Test empty list
    assert list(batch_ids([], 10)) == []

    # Test list smaller than batch size
    ids = ["29110018", "29110015"]
    batches = list(batch_ids(ids, 10))
    assert len(batches) == 1
    assert batches[0] == ids

    # Test list equal to batch size
    ids = ["29110018", "29110015"]
    batches = list(batch_ids(ids, 2))
    assert len(batches) == 1
    assert batches[0] == ids

    # Test list larger than batch size
    ids = ["29110018", "29110015", "29110018", "29110015"]
    batches = list(batch_ids(ids, 2))
    assert len(batches) == 2
    assert batches[0] == ["29110018", "29110015"]
    assert batches[1] == ["29110018", "29110015"]

def test_truncate_values():
    """Test truncate_values function"""
    # Test with short values (no truncation needed)
    xml_short = '<root><Item>Short text</Item></root>'
    result = truncate_values(xml_short, max_length=100)
    assert 'Short text' in result
    assert '...[truncated]' not in result

    # Test with long values (truncation needed)
    long_text = 'x' * 1000
    xml_long = f'<root><Item>{long_text}</Item></root>'
    result = truncate_values(xml_long, max_length=100)
    parsed = ET.fromstring(result)
    item_text = parsed.find('Item').text
    assert '...[truncated]' in item_text
    assert len(item_text) <= 120

    # Test with invalid XML
    result = truncate_values('Not XML', max_length=100)
    assert result == 'Not XML'

    # Test with empty string
    assert truncate_values('', max_length=100) == ''

    # Test with None
    assert truncate_values(None, max_length=100) == None

def test_xml2json():
    """Test xml2json function"""
    # Test valid XML
    xml = '<root><item>Test</item></root>'
    result = xml2json(xml)
    assert isinstance(result, str)
    assert 'root' in result
    assert 'item' in result
    assert 'Test' in result

    # Test invalid XML
    result = xml2json('Not XML')
    assert result == 'Not XML'

    # Test empty string
    assert xml2json('') == ''

    # Test complex XML
    xml = '''
    <root>
        <item id="1">
            <name>Test 1</name>
            <value>100</value>
        </item>
        <item id="2">
            <name>Test 2</name>
            <value>200</value>
        </item>
    </root>
    '''
    result = xml2json(xml)
    assert isinstance(result, str)
    assert 'Test 1' in result
    assert 'Test 2' in result
    
def test_to_json():
    """Test to_json function"""
    from decimal import Decimal
    from datetime import datetime

    class MockResult:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            return iter([self.data])

    # Test with datetime
    dt = datetime(2024, 1, 1, 12, 0)
    data = {'datetime': dt}
    result = MockResult(data)
    json_str = to_json(result)
    assert '"datetime": "2024-01-01T12:00:00"' in json_str

    # Test with Decimal
    dec = Decimal('10.5')
    data = {'decimal': dec}
    result = MockResult(data)
    json_str = to_json(result)
    assert '"decimal": "10.5"' in json_str

    # Test with empty results
    result = MockResult({})
    json_str = to_json(result)
    assert json_str == '[{}]'

    # Test with None values
    data = {'null_value': None}
    result = MockResult(data)
    json_str = to_json(result)
    assert '"null_value": null' in json_str

def test_join_accs():
    """Test join_accs function"""
    # Test empty list
    assert join_accs([]) == ''

    # Test single accession
    assert join_accs(['29110018']) == "'29110018'"

    # Test multiple accessions
    result = join_accs(['29110018', '29110015'])
    assert "'29110018'" in result
    assert "'29110015'" in result
    assert ',' in result

    # Test with duplicates (should preserve duplicates)
    result = join_accs(['29110018', '29110018'])
    assert result == "'29110018', '29110018'"

    # Test with different accession types
    result = join_accs(['SRR123', 'GSE456'])
    assert "'SRR123'" in result
    assert "'GSE456'" in result

def test_run_cmd_with_spaces():
    """Test run_cmd with arguments containing spaces"""
    # Test echo with quoted argument containing spaces
    returncode, output, error = run_cmd(['echo', 'hello there world'])
    assert returncode == 0
    assert output.decode().strip() == 'hello there world'
    assert error == b''