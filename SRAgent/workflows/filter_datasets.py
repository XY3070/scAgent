# import
## batteries
import os
import sys
import asyncio
from typing import List, Dict, Any, Tuple, Annotated, TypedDict, Sequence, Callable, Optional, Union
## 3rd party
import pandas as pd
from pydantic import BaseModel, Field
from langgraph.types import Send
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
## package
from SRAgent.agents.utils import set_model
from SRAgent.db.connect import db_connect

# classes
class FilterCriteria(BaseModel):
    """Filter criteria for dataset search"""
    # Required criteria
    organism: str = Field(
        description="Organism name, e.g., 'homo sapiens'"
    )
    exclude_cell_line: bool = Field(
        description="Whether to exclude cell lines",
        default=True
    )
    data_available: bool = Field(
        description="Whether data is available",
        default=True
    )
    cancer_annotation: bool = Field(
        description="Whether cancer/tumor annotation is required",
        default=True
    )
    sequencing_method: str = Field(
        description="Sequencing method, e.g., '10X genomics', 'smart-seq2'"
    )
    tissue_source: bool = Field(
        description="Whether tissue source is specified",
        default=True
    )
    # Optional criteria
    has_publication: bool = Field(
        description="Whether publication information (PMID or DOI) is required",
        default=False
    )
    has_sample_size: bool = Field(
        description="Whether sample size information is required",
        default=False
    )
    has_nationality: bool = Field(
        description="Whether nationality/region information is required",
        default=False
    )

class FilterResult(BaseModel):
    """Result of dataset filtering"""
    datasets: List[Dict[str, Any]] = Field(
        description="List of datasets matching the filter criteria"
    )
    count: int = Field(
        description="Number of datasets matching the filter criteria"
    )

# functions
def filter_datasets(criteria: FilterCriteria) -> FilterResult:
    """Filter datasets based on criteria"""
    try:
        with db_connect() as conn:
            # Start building the query
            query = """
            SELECT 
                sm.srx_accession, 
                sm.organism, 
                sm.tissue, 
                sm.disease, 
                sm.cell_line,
                sm.database,
                sm.entrez_id,
                sm.tech_10x,
                sm.lib_prep,
                sm.notes,
                sm.publication,
                sm.sample_size,
                sm.nationality,
                ss.srr_accession 
            FROM srx_metadata sm
            LEFT JOIN srx_srr ss ON sm.srx_accession = ss.srx_accession
            WHERE 1=1
            """
            
            # Add filter conditions
            params = {}
            
            # Required criteria
            if criteria.organism:
                query += " AND sm.organism ILIKE %(organism)s"
                params['organism'] = f'%{criteria.organism}%'
            
            if criteria.exclude_cell_line:
                query += " AND (sm.cell_line IS NULL OR sm.cell_line = '' OR sm.cell_line NOT ILIKE '%cell line%')"
            
            if criteria.data_available:
                query += " AND ss.srr_accession IS NOT NULL"
                query += " AND sm.database IS NOT NULL AND sm.entrez_id IS NOT NULL"
            
            if criteria.cancer_annotation:
                query += " AND (sm.disease ILIKE '%cancer%' OR sm.disease ILIKE '%tumor%' OR "
                query += "sm.disease ILIKE '%carcinoma%' OR sm.disease ILIKE '%neoplasm%' OR "
                query += "sm.disease ILIKE '%malignant%' OR sm.disease ILIKE '%sarcoma%')"
            
            if criteria.sequencing_method:
                query += " AND (sm.tech_10x ILIKE %(seq_method)s OR sm.lib_prep ILIKE %(seq_method)s)"
                params['seq_method'] = f'%{criteria.sequencing_method}%'
            
            if criteria.tissue_source:
                query += " AND sm.tissue IS NOT NULL AND sm.tissue != ''"
            
            # Optional criteria
            if criteria.has_publication:
                query += " AND (sm.publication IS NOT NULL AND sm.publication != '')"
                query += " OR (sm.notes ILIKE '%PMID%' OR sm.notes ILIKE '%DOI%')"
            
            # Check for sample size information
            if criteria.has_sample_size:
                query += " AND (sm.sample_size IS NOT NULL AND sm.sample_size != '')"
                query += " OR (sm.notes ILIKE '%sample size%' OR sm.notes ILIKE '%number of samples%')"
            
            # Check for nationality/region information
            if criteria.has_nationality:
                query += " AND (sm.nationality IS NOT NULL AND sm.nationality != '')"
                query += " OR (sm.notes ILIKE '%nationality%' OR sm.notes ILIKE '%region%' OR "
                query += "sm.notes ILIKE '%country%' OR sm.notes ILIKE '%population%')"
            
            # Limit results to a reasonable number
            query += " LIMIT 100"
            
            # Execute the query
            df = pd.read_sql(query, conn, params=params)
            
            # Format the results for better readability
            formatted_datasets = []
            for _, row in df.iterrows():
                dataset = {
                    "srx_accession": row.get("srx_accession"),
                    "srr_accession": row.get("srr_accession"),
                    "database": row.get("database"),
                    "entrez_id": row.get("entrez_id"),
                    "organism": row.get("organism"),
                    "tissue": row.get("tissue"),
                    "disease": row.get("disease"),
                    "sequencing_method": row.get("tech_10x") or row.get("lib_prep"),
                    "cell_line": row.get("cell_line"),
                }
                
                # Add optional information if available
                if row.get("publication"):
                    dataset["publication"] = row.get("publication")
                if row.get("sample_size"):
                    dataset["sample_size"] = row.get("sample_size")
                if row.get("nationality"):
                    dataset["nationality"] = row.get("nationality")
                
                # Add notes if they contain relevant information
                notes = row.get("notes")
                if notes:
                    if criteria.has_publication and ("PMID" in notes or "DOI" in notes):
                        dataset["publication_info"] = notes
                    if criteria.has_sample_size and ("sample size" in notes.lower() or "number of samples" in notes.lower()):
                        dataset["sample_size_info"] = notes
                    if criteria.has_nationality and any(term in notes.lower() for term in ["nationality", "region", "country", "population"]):
                        dataset["nationality_info"] = notes
                
                formatted_datasets.append(dataset)
            
            return FilterResult(datasets=formatted_datasets, count=len(formatted_datasets))
    except Exception as e:
        print(f"Error in filter_datasets: {e}")
        return FilterResult(datasets=[], count=0)

@tool
def filter_datasets_tool(
    organism: str,
    exclude_cell_line: bool = True,
    data_available: bool = True,
    cancer_annotation: bool = True,
    sequencing_method: str = "",
    tissue_source: bool = True,
    has_publication: bool = False,
    has_sample_size: bool = False,
    has_nationality: bool = False,
) -> Dict[str, Any]:
    """Filter datasets based on criteria
    
    Args:
        organism: Organism name, e.g., 'homo sapiens'
        exclude_cell_line: Whether to exclude cell lines
        data_available: Whether data is available
        cancer_annotation: Whether cancer/tumor annotation is required
        sequencing_method: Sequencing method, e.g., '10X genomics', 'smart-seq2'
        tissue_source: Whether tissue source is specified
        has_publication: Whether publication information is required
        has_sample_size: Whether sample size information is required
        has_nationality: Whether nationality/region information is required
    
    Returns:
        A dictionary containing the filtered datasets and count
    """
    try:
        # Validate inputs
        if not organism:
            return {"error": "Organism must be specified", "datasets": [], "count": 0}
        
        # Create filter criteria
        criteria = FilterCriteria(
            organism=organism,
            exclude_cell_line=exclude_cell_line,
            data_available=data_available,
            cancer_annotation=cancer_annotation,
            sequencing_method=sequencing_method,
            tissue_source=tissue_source,
            has_publication=has_publication,
            has_sample_size=has_sample_size,
            has_nationality=has_nationality,
        )
        
        # Apply filters
        result = filter_datasets(criteria)
        
        # Return results
        return {
            "datasets": result.datasets,
            "count": result.count,
            "criteria": {
                "organism": organism,
                "exclude_cell_line": exclude_cell_line,
                "data_available": data_available,
                "cancer_annotation": cancer_annotation,
                "sequencing_method": sequencing_method,
                "tissue_source": tissue_source,
                "has_publication": has_publication,
                "has_sample_size": has_sample_size,
                "has_nationality": has_nationality,
            }
        }
    except Exception as e:
        return {"error": str(e), "datasets": [], "count": 0}

def create_filter_datasets_workflow(
    model_name: Optional[str]=None,
    service_tier: Optional[str]=None,
) -> Callable:
    """Create a workflow for filtering datasets"""
    # create model
    model = set_model(model_name=model_name, agent_name="filter_datasets", service_tier=service_tier)

    # set tools
    tools = [
        filter_datasets_tool,
    ]
  
    # state modifier
    state_mod = """
    # Introduction
    You are a helpful senior bioinformatician assisting a researcher with finding single-cell RNA sequencing datasets based on specific criteria. You have access to a database containing metadata about various datasets from public repositories like GEO and SRA.
    
    # Task
    Your task is to help the user filter datasets based on their requirements. You should understand the user's natural language query, extract the relevant filtering criteria, and use the appropriate tool to search for matching datasets.
    
    # Available Tools
    - filter_datasets_tool: Use this tool to filter datasets based on criteria.
      Parameters:
        - organism: Organism name, e.g., 'homo sapiens' (required)
        - exclude_cell_line: Whether to exclude cell lines (default: True)
        - data_available: Whether data is available (default: True)
        - cancer_annotation: Whether cancer/tumor annotation is required (default: True)
        - sequencing_method: Sequencing method, e.g., '10X genomics', 'smart-seq2' (required)
        - tissue_source: Whether tissue source is specified (default: True)
        - has_publication: Whether publication information is required (default: False)
        - has_sample_size: Whether sample size information is required (default: False)
        - has_nationality: Whether nationality/region information is required (default: False)
    
    # Understanding User Queries
    Users may express their filtering requirements in various ways. Here are some examples:
    - "Find datasets for human brain tissue using 10X genomics"
      → organism="homo sapiens", tissue_source=True, sequencing_method="10X genomics"
    - "I need cancer datasets with publication information"
      → cancer_annotation=True, has_publication=True
    - "Show me smart-seq2 data from mouse samples with sample size information"
      → organism="mouse", sequencing_method="smart-seq2", has_sample_size=True
    
    # Workflow
    1. Carefully analyze the user's query to identify the filtering criteria they're interested in.
    2. For any criteria not explicitly mentioned, use the default values.
    3. Use the filter_datasets_tool to search for datasets matching the criteria.
    4. Present the results to the user in a clear and organized manner.
    
    # Response Format
    Your response should include:
    
    1. A summary of the search criteria used:
       - Organism: [organism name]
       - Exclude cell lines: [Yes/No]
       - Data availability: [Required/Not required]
       - Cancer annotation: [Required/Not required]
       - Sequencing method: [method name]
       - Tissue source: [Required/Not required]
       - Publication information: [Required/Not required]
       - Sample size information: [Required/Not required]
       - Nationality/region information: [Required/Not required]
    
    2. Results summary:
       - Number of datasets found: [count]
       - Brief overview of the types of datasets found
    
    3. Detailed results (if datasets were found):
       - Present the datasets in a table format with columns for:
         * Accession (SRX and SRR)
         * Database and ID
         * Organism
         * Tissue
         * Disease/Cancer type
         * Sequencing method
         * Additional information (publication, sample size, nationality) if requested
    
    4. Suggestions (if few or no datasets were found):
       - Recommend modifications to the search criteria that might yield more results
       - Suggest alternative sequencing methods or organisms that have more data available
    
    # Important Notes
    - Always prioritize accuracy over quantity. It's better to return fewer, highly relevant datasets than many loosely matching ones.
    - If the user's query is ambiguous, ask clarifying questions before performing the search.
    - Remember that the database may not have complete information for all fields, so some datasets might be missing certain metadata.
    - The search is limited to 100 results to ensure reasonable response times.
    """
    
    
    # create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=state_mod,
    )

    # create tool
    @tool
    async def invoke_filter_datasets_workflow(
        messages: Annotated[List[BaseMessage], "Messages to send to the Filter Datasets agent"],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the Filter Datasets agent"]:
        """
        Invoke the Filter Datasets agent with a message.
        The Filter Datasets agent will search for datasets matching the specified criteria.
        """
        try:
            response = await agent.ainvoke({"messages": messages}, config=config)
            return response
        except Exception as e:
            return {"error": str(e)}

    return invoke_filter_datasets_workflow

# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set environment variables
    os.environ["DYNACONF"] = "test"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test")
    
    # Test the workflow
    async def main():
        workflow = create_filter_datasets_workflow(service_tier="qwen")
        query = HumanMessage(content="Find datasets for homo sapiens with 10X genomics sequencing method")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print(result)
    
    asyncio.run(main())