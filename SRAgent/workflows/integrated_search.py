# import
## batteries
import os
import sys
import asyncio
from typing import List, Dict, Any, Tuple, Annotated, TypedDict, Sequence, Callable, Optional, Union
## 3rd party
from pydantic import BaseModel, Field
from langgraph.types import Send
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
## package
from SRAgent.agents.utils import set_model
from SRAgent.workflows.tissue_ontology import create_tissue_ontology_workflow
from SRAgent.workflows.filter_datasets import create_filter_datasets_workflow, FilterCriteria

# functions
def create_integrated_search_workflow(
    model_name: Optional[str]=None,
    service_tier: Optional[str]=None,
) -> Callable:
    """Create a workflow that integrates tissue ontology and dataset filtering"""
    # Create sub-workflows
    tissue_ontology_workflow = create_tissue_ontology_workflow(model_name=model_name, service_tier=service_tier)
    filter_datasets_workflow = create_filter_datasets_workflow(model_name=model_name, service_tier=service_tier)
    
    # Create model
    model = set_model(model_name=model_name, agent_name="integrated_search", service_tier=service_tier)
    
    # Define tools
    @tool
    async def query_tissue_ontology(query: str) -> Dict[str, Any]:
        """Query the tissue ontology to get Uberon IDs for tissue names"""
        try:
            message = HumanMessage(content=f"Tissues: {query}")
            result = await tissue_ontology_workflow.ainvoke({"messages": [message]}, config={})
            return result
        except Exception as e:
            return {"error": str(e)}
    
    @tool
    async def search_datasets(query: str) -> Dict[str, Any]:
        """Search for datasets based on criteria"""
        try:
            message = HumanMessage(content=query)
            result = await filter_datasets_workflow.ainvoke({"messages": [message]}, config={})
            return result
        except Exception as e:
            return {"error": str(e)}
    
    # Define state modifier
    state_mod = """
    # Introduction
    You are a helpful senior bioinformatician assisting a researcher with finding single-cell RNA sequencing datasets. You can help with two main tasks:
    1. Identifying Uberon ontology terms for tissue names
    2. Searching for datasets based on specific criteria
    
    # Available Tools
    - query_tissue_ontology: Get Uberon IDs for tissue names
      Parameters:
        - query: The tissue name to query
    
    - search_datasets: Search for datasets based on criteria
      Parameters:
        - query: A natural language description of the search criteria
    
    # Workflow
    1. Understand the user's request - are they asking about tissue ontology, dataset search, or both?
    2. If the user mentions specific tissues but doesn't explicitly ask for ontology information, you can still use the tissue ontology tool to get more information about the tissues.
    3. For dataset searches, extract the search criteria from the user's query and use the search_datasets tool.
    4. If the user's query involves both tissue identification and dataset search, first use the tissue ontology tool to get information about the tissues, then use that information to enhance the dataset search.
    
    # Response Format
    Your response should be clear, concise, and helpful. Include relevant information from both tools when appropriate, and format the results in a way that's easy for the user to understand.
    
    # Examples
    - If the user asks "What is the Uberon ID for brain cortex?", use the query_tissue_ontology tool.
    - If the user asks "Find datasets for homo sapiens with 10X genomics sequencing", use the search_datasets tool.
    - If the user asks "Find brain cortex datasets for homo sapiens", first use query_tissue_ontology to get information about brain cortex, then use search_datasets to find relevant datasets.
    """
    
    # Create agent
    agent = create_react_agent(
        model=model,
        tools=[query_tissue_ontology, search_datasets],
        prompt=state_mod,
    )
    
    # Create workflow tool
    @tool
    async def invoke_integrated_search_workflow(
        messages: Annotated[List[BaseMessage], "Messages to send to the Integrated Search agent"],
        config: RunnableConfig,
    ) -> Annotated[dict, "Response from the Integrated Search agent"]:
        """Invoke the Integrated Search agent with a message."""
        try:
            response = await agent.ainvoke({"messages": messages}, config=config)
            return response
        except Exception as e:
            return {"error": str(e)}
    
    return invoke_integrated_search_workflow

# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set environment variables
    os.environ["DYNACONF"] = "test"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    
    # Test the workflow
    async def main():
        workflow = create_integrated_search_workflow(service_tier="qwen")
        
        # Test tissue ontology query
        print("\nTesting tissue ontology query...")
        query = HumanMessage(content="What is the Uberon ID for brain cortex?")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print(result)
        
        # Test dataset search
        print("\nTesting dataset search...")
        query = HumanMessage(content="Find datasets for homo sapiens with 10X genomics sequencing method")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print(result)
        
        # Test integrated search
        print("\nTesting integrated search...")
        query = HumanMessage(content="Find brain cortex datasets for homo sapiens using 10X genomics")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print(result)
    
    asyncio.run(main())