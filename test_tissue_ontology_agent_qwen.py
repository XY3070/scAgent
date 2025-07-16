import os
import asyncio
from dotenv import load_dotenv
from SRAgent.agents.tissue_ontology import create_tissue_ontology_agent

# 加载环境变量
load_dotenv(override=True)

# 设置环境变量
os.environ["DYNACONF"] = "test"
os.environ["OPENAI_API_KEY"] = "dummy-key"  # 设置一个虚拟的API密钥

async def main():
    print("Creating tissue_ontology agent...")
    
    # 使用本地部署的Qwen模型创建代理
    agent = create_tissue_ontology_agent(model_name="Qwen3-235B-A22B", service_tier="default")
    print("Agent created successfully!")
    
    # 测试代理
    print("\nAgent testing - query: 'brain cortex'")
    try:
        result = await agent.ainvoke({"tissue_description": "brain cortex"})
        print("\nQuery result:")
        print(result)
    except Exception as e:
        print(f"Agent testing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())