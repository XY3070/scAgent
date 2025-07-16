import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from SRAgent.workflows.integrated_search import create_integrated_search_workflow

# 加载环境变量
load_dotenv(override=True)

# 设置环境变量
os.environ["DYNACONF"] = "test"
os.environ["OPENAI_API_KEY"] = "dummy-key"  # 设置一个虚拟的API密钥

# 数据库连接信息
os.environ["DB_HOST"] = "10.28.1.24"
os.environ["DB_PORT"] = "5432"
os.environ["DB_USER"] = "yanglab"
os.environ["GCP_SQL_DB_PASSWORD"] = "labyang"
os.environ["DB_NAME"] = "screcounter"  # 使用正确的数据库名称

async def main():
    print("Creating integrated_search workflow...")
    
    # 使用本地部署的Qwen模型创建工作流
    workflow = create_integrated_search_workflow(model_name="Qwen3-235B-A22B", service_tier="qwen")
    print("Workflow created successfully!")
    
    # 测试组织本体查询
    print("\nTesting tissue ontology query...")
    try:
        query = HumanMessage(content="What is the Uberon ID for brain cortex?")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Workflow testing failed: {e}")
    
    # 测试数据集搜索
    print("\nTesting dataset search...")
    try:
        query = HumanMessage(content="Find datasets for homo sapiens with 10X genomics sequencing method")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Workflow testing failed: {e}")
    
    # 测试集成搜索
    print("\nTesting integrated search...")
    try:
        query = HumanMessage(content="Find brain cortex datasets for homo sapiens using 10X genomics")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Workflow testing failed: {e}")
    
    # 测试复杂查询
    print("\nTesting complex query...")
    try:
        query = HumanMessage(content="""Find datasets with the following criteria:
        1. Organism: homo sapiens
        2. Tissue: liver
        3. Exclude cell lines
        4. Data must be available
        5. Must have cancer annotation
        6. Sequencing method: smart-seq2
        7. Should have publication information""")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Workflow testing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())