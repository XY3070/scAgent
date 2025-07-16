import os
import asyncio
from dotenv import load_dotenv
from SRAgent.workflows.tissue_ontology import create_tissue_ontology_workflow

# 加载环境变量
load_dotenv(override=True)

# 设置环境变量
os.environ["DYNACONF"] = "test"
os.environ["OPENAI_API_KEY"] = "dummy-key"  # 设置一个虚拟的API密钥

async def main():
    print("Creating tissue_ontology workflow...")
    
    # 使用本地部署的Qwen模型创建工作流
    workflow = create_tissue_ontology_workflow(model_name="Qwen3-235B-A22B", service_tier="default")
    print("Workflow created successfully!")
    
    # 测试工作流
    print("\nWorkflow testing - query: 'brain cortex'")
    try:
        from langchain_core.messages import HumanMessage
        result = await workflow.ainvoke({"messages": [HumanMessage(content="Tissues: brain cortex")]})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Workflow testing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())