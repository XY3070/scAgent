# import
## batteries
import os
import re
from typing import Dict, Any, List, Optional
## 3rd party
from pydantic import BaseModel, Field
from langchain_core.tools import tool
## package
from SRAgent.agents.utils import set_model

# classes
class FilterCriteriaExtraction(BaseModel):
    """Extracted filter criteria from user query"""
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

# functions
@tool
def create_filter_datasets_agent(
    model_name: Optional[str] = None,
    service_tier: Optional[str] = None,
):
    """Create an agent that extracts filter criteria from user queries"""
    # create model
    model = set_model(model_name=model_name, agent_name="filter_datasets_extraction", service_tier=service_tier)
    
    # create the extraction function
    async def extract_filter_criteria(query: str) -> FilterCriteriaExtraction:
        """Extract filter criteria from user query"""
        prompt = """
        You are a helpful assistant for a bioinformatics researcher.
        Your task is to extract dataset filter criteria from the user's query.
        
        # Filter Criteria to Extract
        1. Organism: The organism name (e.g., 'homo sapiens', 'mouse', 'rat')
        2. Exclude cell line: Whether to exclude cell lines (default: True)
        3. Data available: Whether data should be available (default: True)
        4. Cancer annotation: Whether cancer/tumor annotation is required (default: True)
        5. Sequencing method: The sequencing method (e.g., '10X genomics', 'smart-seq2')
        6. Tissue source: Whether tissue source should be specified (default: True)
        7. Has publication: Whether publication information (PMID or DOI) is required (default: False)
        8. Has sample size: Whether sample size information is required (default: False)
        9. Has nationality: Whether nationality/region information is required (default: False)
        
        # User Query
        {query}
        
        # Instructions
        - Extract all criteria mentioned in the query
        - Use default values for criteria not mentioned
        - For sequencing method, extract the specific method mentioned (e.g., '10X genomics', 'smart-seq2')
        - For organism, extract the specific organism mentioned (e.g., 'homo sapiens', 'mouse', 'rat')
        - Return the extracted criteria in a structured format
        """
        
        # Replace placeholder with actual query
        prompt = prompt.format(query=query)
        
        # Invoke model with structured output
        try:
            response = await model.with_structured_output(FilterCriteriaExtraction).ainvoke(prompt)
            return response
        except Exception as e:
            # Default values if extraction fails
            return FilterCriteriaExtraction(
                organism="homo sapiens",
                exclude_cell_line=True,
                data_available=True,
                cancer_annotation=True,
                sequencing_method="10X genomics",
                tissue_source=True,
                has_publication=False,
                has_sample_size=False,
                has_nationality=False
            )
    
    return extract_filter_criteria

# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import asyncio
    
    async def main():
        agent = create_filter_datasets_agent()
        query = "Find datasets for homo sapiens with 10X genomics sequencing method"
        result = await agent(query)
        print(result)
    
    asyncio.run(main())