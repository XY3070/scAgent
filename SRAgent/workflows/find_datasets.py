#!/usr/bin/env python
# import
import os
import sys
import asyncio
import operator
from typing import List, Dict, Any, Tuple, Annotated, TypedDict, Sequence, Callable
## 3rd party
from dotenv import load_dotenv
import pandas as pd
from Bio import Entrez
from pydantic import BaseModel
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
## package
from SRAgent.agents.utils import set_model
from SRAgent.agents.find_datasets import create_find_datasets_agent
from SRAgent.workflows.srx_info import create_SRX_info_graph
from SRAgent.db.connect import db_connect
from SRAgent.db.upsert import db_upsert
from SRAgent.db.get import db_get_entrez_ids, get_prefiltered_datasets_from_local_db
from SRAgent.config import (
    ENTREZ_ID_EXTRACTION_PROMPT_PREFIX,
    ENTREZ_ID_EXTRACTION_PROMPT_TASKS,
    ENTREZ_ID_EXTRACTION_PROMPT_MESSAGE_START,
    ENTREZ_ID_EXTRACTION_PROMPT_MESSAGE_END,
    ENTREZ_ID_EXTRACTION_PROMPT_RETRY_SUFFIX
)

# state
class GraphState(TypedDict):
    """
    Shared state of the agents in the graph
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    entrez_ids: Annotated[List[int], "List of dataset Entrez IDs"]
    database: Annotated[str, operator.add]  # Database name (e.g., 'sra', 'gds')
    cli_args: Any # 添加cli_args字段，用于传递CLI参数

# nodes
def create_find_datasets_node():
    # create the agent
    agent = create_find_datasets_agent()

    # create the node function
    async def invoke_find_datasets_agent_node(
        state: GraphState,
        config: RunnableConfig,
        ) -> Dict[str, Any]:
        """Invoke the find_datasets agent to get datasets to process"""
        # call the agent
        response = await agent.ainvoke({"message": state["messages"][-1].content}, config=config)
        # return the last message in the response
        return {
            "messages" : [response["messages"][-1]],
        }
    return invoke_find_datasets_agent_node

## entrez IDs extraction
class EntrezInfo(BaseModel):
    entrez_ids: List[int]
    database: str

def create_get_entrez_ids_node() -> Callable:
    model = set_model(agent_name="get_entrez_ids")
    async def invoke_get_entrez_ids_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Structured data extraction of Entrez IDs from message
        """
        # create prompt
        message = state["messages"][-1].content
        prompt = "\n".join([
            ENTREZ_ID_EXTRACTION_PROMPT_PREFIX,
            ENTREZ_ID_EXTRACTION_PROMPT_TASKS,
            ENTREZ_ID_EXTRACTION_PROMPT_MESSAGE_START,
            message,
            ENTREZ_ID_EXTRACTION_PROMPT_MESSAGE_END
        ])
        
        # invoke model with structured output; try 3 times to get valid output
        entrez_ids = []
        max_retries = 3
        database = ""
        for i in range(max_retries):
            try:
                response = await model.with_structured_output(EntrezInfo, strict=True).ainvoke(prompt)
                entrez_ids = response.entrez_ids
                database = str(response.database).lower()
                if database in ["sra", "gds"]:
                    break
            except Exception as e:
                if "OpenAIRefusalError" in str(type(e).__name__) and i < max_retries - 1:
                    print(f"OpenAI refused to extract Entrez IDs (attempt {i + 1}), retrying...", file=sys.stderr)
                    prompt += f"\n{ENTREZ_ID_EXTRACTION_PROMPT_RETRY_SUFFIX}"
                    continue
                else:
                    # For final attempt or other errors, use empty values
                    print(f"Error extracting Entrez IDs: {str(e)}", file=sys.stderr)
                    entrez_ids = []
                    database = ""
                    break
                    
        ## if no valid database, return no entrez IDs
        if database not in ["sra", "gds"]:
            return {"entrez_ids": [], "database": ""}

        # entrez ID check
        ## filter out entrez IDs that are already in the database
        if config.get("configurable", {}).get("use_database"):
            with db_connect() as conn:
                existing_ids = db_get_entrez_ids(conn=conn, database=database)
                entrez_ids = [x for x in entrez_ids if x not in existing_ids]

        # cap number of entrez IDs to max_datasets in config
        max_datasets = config.get("configurable", {}).get("max_datasets")
        if entrez_ids and max_datasets and max_datasets > 0 and len(entrez_ids) > max_datasets:
            entrez_ids = entrez_ids[:max_datasets]

        ## update the database
        if len(entrez_ids) > 0 and config.get("configurable", {}).get("use_database"):
            df = pd.DataFrame({
                "entrez_id": entrez_ids,
                "database": database,
                "notes": "New dataset found by Find-Datasets agent"
            })
            if config.get("configurable", {}).get("use_database"):
                with db_connect() as conn:
                    db_upsert(df, "srx_metadata", conn=conn)

        # return the extracted values
        return {"entrez_ids": entrez_ids, "database": database}
    return invoke_get_entrez_ids_node

def continue_to_srx_info(state: GraphState, config: RunnableConfig) -> List[Dict[str, Any]]:
    """
    Parallel invoke of the srx_info graph
    """
    ## submit each SRX accession to the metadata graph
    responses = []
    # 确保 entrez_ids 和 database 存在且有效
    if "entrez_ids" in state and state["entrez_ids"] and "database" in state and state["database"]:
        for entrez_id in state["entrez_ids"]:
            input = {
                "database": state["database"],
                "entrez_id": str(entrez_id), # 确保 entrez_id 是字符串类型
                "messages": [HumanMessage(content=f"Process Entrez ID {entrez_id} from {state['database']} database.")]
            }
            responses.append(Send("srx_info_node", input))
    return responses

def final_state(state: GraphState) -> Dict[str, Any]:
    """
    Return the final state of the graph
    Args:
        state: The final state of the graph
    Returns:
        The final state of the graph
    """
    # filter to messages that contain the SRX accession
    messages = []
    for msg in state["messages"]:
        try:
            msg = [msg.content]
        except AttributeError:
            msg = [x.content for x in msg]
        for x in msg:
            if x.startswith("# SRX accession: "):
                messages.append(x)
    # filter to unique messages
    messages = list(set(messages))
    # final message
    if len(messages) == 0:
        message = "No novel SRX accessions found."
    else:
        message = "\n".join(messages)
    return {
        "messages": [AIMessage(content=message)]
    }

def get_prefiltered_datasets_node(conn):
    async def invoke_get_prefiltered_datasets_node(
        state: GraphState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        cli_args = state["cli_args"]
        # 调用预筛选函数
        prefiltered_datasets = await get_prefiltered_datasets_from_local_db(
            conn=conn,
            organisms=cli_args.organisms,
            min_date=cli_args.min_date,
            max_date=cli_args.max_date,
            search_term=cli_args.message.strip('"'), # 移除消息中的引号
            limit=cli_args.max_datasets
        )

        # 从DataFrame中提取srx_id作为entrez_ids
        entrez_ids = []
        if prefiltered_datasets:
            # 确保srx_id是整数，并且不为None
            entrez_ids = [int(d['srx_id']) for d in prefiltered_datasets if d.get('srx_id') is not None]

        print(f"Found {len(entrez_ids)} prefiltered SRX IDs.", file=sys.stderr)

        return {"entrez_ids": entrez_ids, "database": "sra"} # 假设数据库是sra
    return invoke_get_prefiltered_datasets_node

def create_local_db_find_datasets_graph(conn):
    #-- graph --#
    workflow = StateGraph(GraphState)

    # nodes
    workflow.add_node("get_prefiltered_datasets_node", get_prefiltered_datasets_node(conn))
    workflow.add_node("srx_info_node", create_SRX_info_graph())
    workflow.add_node("final_state_node", final_state)

    # edges
    workflow.add_edge(START, "get_prefiltered_datasets_node")
    workflow.add_conditional_edges(
        "get_prefiltered_datasets_node",
        lambda state: "final_state_node" if not state["entrez_ids"] else "srx_info_node",
        {"srx_info_node": "srx_info_node", "final_state_node": "final_state_node"}
    )
    workflow.add_edge("srx_info_node", "final_state_node")
    workflow.add_edge("final_state_node", END)

    return workflow.compile()

def get_prefiltered_datasets_node(conn):
    async def invoke_get_prefiltered_datasets_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        cli_args = state["cli_args"]
        prefiltered_datasets = get_prefiltered_datasets_from_local_db(
            conn=conn,
            search_term=state["messages"][-1].content, # Use search_term instead of message
            organisms=cli_args.organisms,
            min_date=cli_args.min_date,
            max_date=cli_args.max_date,
            limit=cli_args.max_datasets # Use cli_args.max_datasets as limit
        )

        # Ensure prefiltered_datasets is a list, even if empty
        if prefiltered_datasets is None:
            prefiltered_datasets = []

        # Extract SRX IDs from the prefiltered_datasets (assuming it's a list of dicts)
        # Filter out any dictionaries that do not have 'srx_id'
        srx_ids = [d['srx_id'] for d in prefiltered_datasets if 'srx_id' in d and d['srx_id'] is not None]

        # Use db_find_srx to get Entrez IDs for these SRX IDs
        # db_find_srx returns a DataFrame, so we need to extract entrez_id from it
        if srx_ids:
            print(f"SRX IDs found: {srx_ids}", file=sys.stderr)
            srx_info_df = db_find_srx(srx_ids, conn)
            if not srx_info_df.empty:
                # Debugging: Print columns and head of srx_info_df
                print(f"SRX Info DataFrame Columns: {srx_info_df.columns.tolist()}", file=sys.stderr)
                print(f"SRX Info DataFrame Head:\n{srx_info_df.head()}", file=sys.stderr)
                # Convert Entrez IDs to integers as expected by GraphState
                if 'entrez_id' in srx_info_df.columns:
                    entrez_ids = srx_info_df['entrez_id'].dropna().astype(int).tolist()
                else:
                    print("Error: 'entrez_id' column not found in srx_info_df", file=sys.stderr)
                    entrez_ids = []
            else:
                print("db_find_srx returned an empty DataFrame.", file=sys.stderr)
                entrez_ids = []
        else:
            print("No SRX IDs extracted from prefiltered_datasets.", file=sys.stderr)
            entrez_ids = []

        return {"entrez_ids": entrez_ids, "database": "sra"}
    return invoke_get_prefiltered_datasets_node

def create_find_datasets_graph():
    #-- graph --#
    workflow = StateGraph(GraphState)

    # nodes
    workflow.add_node("search_datasets_node", create_find_datasets_node())
    workflow.add_node("get_entrez_ids_node", create_get_entrez_ids_node())
    workflow.add_node("srx_info_node", create_SRX_info_graph())
    workflow.add_node("final_state_node", final_state)

    # edges
    workflow.add_edge(START, "search_datasets_node")
    workflow.add_edge("search_datasets_node", "get_entrez_ids_node")
    workflow.add_conditional_edges(
        "get_entrez_ids_node",
        lambda state: "final_state_node" if not state["entrez_ids"] else "srx_info_node",
        {"srx_info_node": "srx_info_node", "final_state_node": "final_state_node"}
    )
    workflow.add_edge("srx_info_node", "final_state_node")
    workflow.add_edge("final_state_node", END)

    # compile the graph
    graph = workflow.compile()
    return graph

def create_local_db_find_datasets_graph(conn):
    #-- graph --#
    workflow = StateGraph(GraphState)

    # nodes
    workflow.add_node("get_prefiltered_datasets_node", get_prefiltered_datasets_node)
    workflow.add_node("get_entrez_ids_node", create_get_entrez_ids_node())
    workflow.add_node("srx_info_node", create_SRX_info_graph())
    workflow.add_node("final_state_node", final_state)

    # edges
    workflow.add_edge(START, "get_prefiltered_datasets_node")
    workflow.add_edge("get_prefiltered_datasets_node", "get_entrez_ids_node")
    workflow.add_conditional_edges(
        "get_entrez_ids_node",
        lambda state: "final_state_node" if not state["entrez_ids"] else "srx_info_node",
        {"srx_info_node": "srx_info_node", "final_state_node": "final_state_node"}
    )
    workflow.add_edge("srx_info_node", "final_state_node")
    workflow.add_edge("final_state_node", END)

    # compile the graph
    graph = workflow.compile()
    return graph

# 新的节点 1: SQL 预筛选节点
async def sql_prefilter_node(state: dict, conn) -> dict:
    """Node to perform SQL pre-filtering on the local database."""
    message = state["messages"][-1].content # 从messages中获取最新消息作为搜索词
    args = state["cli_args"] # 假设我们将CLI参数放入state
    
    prefiltered_results = await get_prefiltered_datasets_from_local_db(
        conn=conn,
        organisms=args.organisms,
        min_date=args.min_date,
        max_date=args.max_date,
        search_term=message, # 将整个用户消息作为搜索词
        limit=100 # 设置一个合理的预筛选上限
    )
    
    return {"prefiltered_data": prefiltered_results, "messages": state["messages"] + [AIMessage(content="SQL pre-filtering complete.")]}

# 新的节点 2: LLM 二次精筛节点
def create_llm_refine_node(model):
    """Creates a node for LLM to refine the pre-filtered dataset list."""
    async def invoke_llm_refine_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        # 这是一个占位符实现，实际的LLM精筛逻辑需要在这里实现
        # 假设它只是简单地返回预筛选的数据
        print("Invoking LLM refine node (placeholder)", file=sys.stderr)
        return {"messages": state["messages"] + [AIMessage(content="LLM refine complete.")]}
    return invoke_llm_refine_node

# 新的工作流构建函数
def create_local_db_find_datasets_graph(conn):
    """Creates the LangGraph workflow for finding datasets from the local DB."""
    workflow = StateGraph(GraphState)

    # 定义节点
    async def invoke_sql_prefilter_node(state: dict) -> dict:
        return await sql_prefilter_node(state, conn)
    workflow.add_node("sql_prefilter", invoke_sql_prefilter_node)
    
    llm_refine_node = create_llm_refine_node(set_model(agent_name="llm_refine")) # 假设llm_refine也需要一个模型
    workflow.add_node("llm_refine", llm_refine_node)
    srx_info_node = create_SRX_info_graph() # 可以复用
    workflow.add_node("process_srx_info", srx_info_node)
    workflow.add_node("final_state_node", final_state)
    
    # 定义流程
    workflow.set_entry_point("sql_prefilter")
    workflow.add_edge("sql_prefilter", "llm_refine")
    workflow.add_edge("llm_refine", "process_srx_info") # 暂时直接连接，后续可能需要条件边
    workflow.add_edge("process_srx_info", "final_state_node")
    workflow.add_edge("final_state_node", END)
    
    return workflow.compile()


# main
if __name__ == "__main__":
    from Bio import Entrez

    #-- setup --#
    from dotenv import load_dotenv
    load_dotenv(override=True)
    Entrez.email = os.getenv("EMAIL")

    #-- graph --#
    async def main():
        # <--- MODIFICATION START --->
        # 将CLI参数也放入初始状态，便于本地工作流节点访问
        # 对于本地工作流，message是search_term；对于API工作流，是给Agent的指令
        initial_state = {
            "cli_args": args # 将args对象直接传入，方便节点访问
        }
        if args.source == 'local':
            initial_state["messages"] = [HumanMessage(content=args.message)]
            initial_state["entrez_ids"] = [int(args.message)] if args.message.isdigit() else []
            initial_state["database"] = "sra" # 假设本地模式下默认是sra数据库
        else:
            initial_state["messages"] = [HumanMessage(content=args.message)]
        # <--- MODIFICATION END --->
        config = {"max_concurrency" : 4, "recursion_limit": 200, "configurable": {"organisms": ["rat"]}}
        graph = create_find_datasets_graph()
        async for step in graph.astream(input, config=config):
            print(step)
    asyncio.run(main())