import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from SRAgent.workflows.filter_datasets import create_filter_datasets_workflow

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
os.environ["DB_NAME"] = "scagent"  # 使用正确的数据库名称

async def main():
    print("Creating filter_datasets workflow...")
    
    # 使用本地部署的Qwen模型创建工作流
    workflow = create_filter_datasets_workflow(model_name="Qwen3-235B-A22B", service_tier="qwen")
    print("Workflow created successfully!")
    
    # 测试工作流
    print("\nWorkflow testing - query: 'Find datasets for homo sapiens with 10X genomics sequencing method'")
    try:
        query = HumanMessage(content="Find datasets for homo sapiens with 10X genomics sequencing method")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Workflow testing failed: {e}")

    # 测试更复杂的查询
    print("\nWorkflow testing - complex query")
    try:
        query = HumanMessage(content="""Find datasets with the following criteria:
        1. Organism: homo sapiens
        2. Exclude cell lines
        3. Data must be available
        4. Must have cancer annotation
        5. Sequencing method: smart-seq2
        6. Must have tissue source information
        7. Should have publication information""")
        result = await workflow.ainvoke({"messages": [query]}, config={})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Workflow testing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())