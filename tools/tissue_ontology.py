# import
## batteries
import os
import re
import sys
import time
import shutil
import tarfile
import tempfile
import requests
import urllib.parse
from typing import Annotated
from functools import lru_cache
## 3rd party
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import obonet
import networkx as nx
import appdirs


# functions
def verify_collection(persistent_client: chromadb.PersistentClient, collection_name: str) -> None:
    """
    Verify that the collection exists and has documents
    Args:
        persistent_client: The persistent Chroma client
        collection_name: The name of the collection to verify
    Returns:
        None
    Raises:
        Exception: If the collection does not exist or has no documents
    """
    try:
        collection = persistent_client.get_collection(collection_name)
        count = collection.count()
        print(f"Found {count} documents in collection '{collection_name}'", file=sys.stdout)
    except Exception as e:
        msg = f"Error accessing collection: {e}"
        msg += f"\nAvailable collections: {persistent_client.list_collections()}"
        raise Exception(msg)

def load_vector_store(chroma_path: str, collection_name: str="uberon") -> Chroma:
    """
    Load a Chroma vector store from the specified path.
    Args:
        chroma_path: The path to the Chroma DB directory
        collection_name: The name of the collection to load
    Returns:
        A Chroma vector store
    Raises:
        FileNotFoundError: If the Chroma DB directory does not exist
        Exception: If the collection does not exist or has no documents
    """
    # Ensure the path exists
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"Chroma DB directory not found: {chroma_path}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load the persistent Chroma client
    persistent_client = chromadb.PersistentClient(path=chroma_path)
    
    # Load the existing vector store
    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vector_store

@tool
def query_vector_db(
    query: Annotated[str, "The semantic search query"],
    k: Annotated[int, "The number of results to return"]=3,
    ) -> str: 
    """
    Perform a semantic search by querying a vector store 
    """
    # Determine the cache directory using appdirs
    cache_dir = appdirs.user_cache_dir("SRAgent")
    chroma_dir_name = "uberon-full_chroma"
    chroma_dir_path = os.path.join(cache_dir, chroma_dir_name)
    
    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if the Chroma DB directory exists
    if not os.path.exists(chroma_dir_path) or not os.listdir(chroma_dir_path):
        # Download and extract the tarball
        gcp_url = "https://storage.googleapis.com/arc-scbasecount/2025-02-25/tissue_ontology/uberon-full_chroma.tar.gz"
        tarball_path = os.path.join(cache_dir, "uberon-full_chroma.tar.gz")
        
        print(f"Downloading Chroma DB from {gcp_url}...", file=sys.stdout)
        try:
            # Download the tarball
            response = requests.get(gcp_url, stream=True)
            response.raise_for_status()
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded to {tarball_path}, extracting...", file=sys.stdout)
            
            # Extract the tarball
            with tarfile.open(tarball_path) as tar:
                # Create a temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    tar.extractall(path=temp_dir)
                    # Find the extracted directory
                    extracted_items = os.listdir(temp_dir)
                    if extracted_items:
                        extracted_dir = os.path.join(temp_dir, extracted_items[0])
                        # If the extraction created a directory with the right name, move it
                        if os.path.isdir(extracted_dir):
                            # Remove any existing directory
                            if os.path.exists(chroma_dir_path):
                                shutil.rmtree(chroma_dir_path)
                            # Move the extracted directory to the cache
                            shutil.move(extracted_dir, chroma_dir_path)
            
            print(f"Extraction complete, Chroma DB available at {chroma_dir_path}", file=sys.stdout)
            
            # Clean up the downloaded tarball
            os.remove(tarball_path)
        except Exception as e:
            return f"Error downloading or extracting Chroma DB: {e}"
    
    # Load the vector store
    vector_store = load_vector_store(chroma_dir_path)
    
    # Query the vector store
    message = ""
    try:
        results = vector_store.similarity_search(query, k=k)
        message += f"# Results for query: \"{query}\"\n"
        for i, res in enumerate(results, 1):
            id = res.metadata.get("id", "No ID available")
            if not id:
                continue
            message += f"{i}. {id}\n"
            name = res.metadata.get("name", "No name available")
            message += f"  Ontology name: {name}\n"
            message += f"  Description: {res.page_content}\n"
    except Exception as e:
        return f"Error performing search: {e}"
    if not message:
        message = f"No results found for query: \"{query}\". Consider refining your query."
    return message

# Cache for the ontology graph
_ONTOLOGY_GRAPH = None

@lru_cache(maxsize=1)
def get_ontology_graph(obo_path: str) -> nx.MultiDiGraph:
    """
    Load and cache the ontology graph from the OBO file.
    Uses lru_cache to ensure the graph is only loaded once.
    Args:
        obo_path: Path to the OBO file
    Returns:
        The ontology graph as a NetworkX MultiDiGraph
    """
    return obonet.read_obo(obo_path)

@tool 
def get_neighbors(
    uberon_id: Annotated[str, "The Uberon ID (UBERON:XXXXXXX)"],
    ) -> str: 
    """
    Get the neighbors of a given Uberon ID in the Uberon tissue ontology.
    """
    # check the ID format
    if not re.match(r"UBERON:\d{7}", uberon_id):
        return f"Invalid Uberon ID format: \"{uberon_id}\". The format must be \"UBERON:XXXXXXX\"."

    # Determine the cache directory using appdirs
    cache_dir = appdirs.user_cache_dir("SRAgent")
    obo_filename = "uberon-full.obo"
    obo_path = os.path.join(cache_dir, obo_filename)
    
    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download the OBO file if it doesn't exist
    if not os.path.exists(obo_path) or not os.listdir(cache_dir):
        obo_url = "http://purl.obolibrary.org/obo/uberon/uberon-full.obo"
        print(f"Downloading Uberon ontology from {obo_url}...", file=sys.stdout)
        try:
            response = requests.get(obo_url)
            response.raise_for_status()
            with open(obo_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded and saved to {obo_path}", file=sys.stdout)
        except Exception as e:
            return f"Error downloading Uberon ontology: {e}"

    # Get the cached ontology graph or load it if not available
    g = get_ontology_graph(obo_path)

    # get neighbors
    message = ""
    try:
        message += f"# Neighbors in the ontology for: \"{uberon_id}\"\n"
        for i,node_id in enumerate(g.neighbors(uberon_id), 1):
            # filter out non-UBERON nodes
            if not node_id.startswith("UBERON:") or not g.nodes[node_id]:
                continue
            # extract node name and description
            node_name = g.nodes[node_id]["name"]
            node_def = g.nodes[node_id]["def"]
            message += f"{i}. {node_id}\n"
            message += f"  Ontology name: {node_name}\n"
            message += f"  Description: {node_def}\n"
            # limit to 50 neighbors
            if i >= 50:
                break
    except Exception as e:
        return f"Error getting neighbors: {e}"

    if not message:
        message = f"No neighbors found for ID: \"{uberon_id}\"."
    return message

@tool
def query_uberon_ols(
    search_term: Annotated[str, "The term to search for in the Uberon ontology"]
) -> str:
    """
    Query the Ontology Lookup Service (OLS) for Uberon terms matching the search term.
    """
    # Format search term for URL (handle special characters)
    import urllib.parse
    encoded_search_term = urllib.parse.quote(search_term)
    #print(f"Encoded search term: {encoded_search_term}"); exit();
    
    url = f"https://www.ebi.ac.uk/ols/api/search?q={encoded_search_term}&ontology=uberon"
    max_retries = 2
    retry_delay = 1
    
    for retry in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            return f"Error querying OLS API after {max_retries} attempts: {e}"

    results = data.get("response", {}).get("docs", [])
    if not results:
        return f"No results found for search term: '{search_term}'."

    message = f"# Results from OLS for '{search_term}':\n"
    for i, doc in enumerate(results, 1):
        # Each doc should have an 'obo_id', a 'label', and possibly a 'description'
        obo_id = doc.get("obo_id", "No ID")
        if not obo_id.startswith("UBERON:"):
            continue
        label = doc.get("label", "No label")
        description = doc.get("description", ["None provided"])
        try:
            description = description[0]
        except IndexError:
            pass
        if not description:
            description = "None provided"
        # print description class
        message += f"{i}. {obo_id} - {label}\n   Description: {description}\n"
    return message

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # semantic search
    query = "brain"
    results = query_vector_db.invoke({"query" : query})
    print(results); exit();

    #  get neighbors
    input = {'uberon_id': 'UBERON:0000010'}
    #input = {'uberon_id': 'UBERON:0002421'}
    neighbors = get_neighbors.invoke(input)
    print(neighbors); exit();

    # query OLS
    input = {'search_term': "bone marrow"}
    results = query_uberon_ols.invoke(input)
    print(results)
    

    
