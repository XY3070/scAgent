# 在SRAgent中集成本地部署的Qwen模型

本指南介绍如何在SRAgent项目中集成和使用本地部署的Qwen模型。

## 1. 修改utils.py文件

首先，需要修改`SRAgent/agents/utils.py`文件，添加对Qwen模型的支持。在`set_model`函数中添加以下代码：

```python
elif model_name.startswith("Qwen"):
    # 创建自定义请求头
    default_headers = {}
    if reasoning_effort is not None:
        default_headers["X-Qwen-Enable-Thinking"] = "true"
    else:
        default_headers["X-Qwen-Enable-Thinking"] = "false"
        
    model = ChatOpenAI(
        model_name=model_name,
        openai_api_base="http://10.28.1.21:30080/v1",
        openai_api_key="dummy-key",  # API不需要真实的key，但需要提供一个值
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.8,
        presence_penalty=1.5,
        default_headers=default_headers
    )
```

## 2. 修改settings.yml文件

在`settings.yml`文件中，为需要使用Qwen模型的代理添加配置：

```yaml
test:
  models:
    tissue_ontology: "Qwen3-235B-A22B"  # 使用本地部署的Qwen模型
  temperature:
    tissue_ontology: 0.7  # 为Qwen模型设置更高的温度
  reasoning_effort:
    tissue_ontology: null  # Qwen模型不使用reasoning_effort
  service_tier:
    tissue_ontology: "default"  # Qwen模型不使用flex服务层级
```

## 3. 修改代理函数

确保代理创建函数支持service_tier参数：

```python
def create_tissue_ontology_agent(
    model_name: Optional[str]=None,
    return_tool: bool=True,
    service_tier: Optional[str]=None,
) -> Callable:
    # create model
    model = set_model(model_name=model_name, agent_name="tissue_ontology", service_tier=service_tier)
    # ...
```

## 4. 使用示例

### 直接API调用

```python
import requests
import json

# 使用thinking模式
url = "http://10.28.1.21:30080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "Qwen3-235B-A22B",
    "messages": [
        {"role": "user", "content": "Give me a short introduction to large language models."}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 1000
}

response = requests.post(url, headers=headers, json=data, timeout=120)
result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))

# 使用no-thinking模式
headers = {
    "Content-Type": "application/json",
    "X-Qwen-Enable-Thinking": "false"
}
data = {
    "model": "Qwen3-235B-A22B",
    "messages": [
        {"role": "user", "content": "Give me a short introduction to large language models."}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 1000,
    "presence_penalty": 1.5
}

response = requests.post(url, headers=headers, json=data, timeout=120)
result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### 通过LangChain集成

```python
from langchain_core.messages import HumanMessage
from SRAgent.agents.utils import set_model

# 创建模型
model = set_model(model_name="Qwen3-235B-A22B", temperature=0.7, reasoning_effort=None, max_tokens=1000)

# 创建消息
message = HumanMessage(content="Give me a short introduction to large language models.")

# 调用模型
response = model.invoke([message])
print(response.content)
```

### 使用tissue_ontology代理

```python
import os
import asyncio
from SRAgent.agents.tissue_ontology import create_tissue_ontology_agent

# 设置环境变量
os.environ["DYNACONF"] = "test"

async def main():
    # 创建代理
    agent = create_tissue_ontology_agent(model_name="Qwen3-235B-A22B", service_tier="default")
    
    # 调用代理
    result = await agent.ainvoke({"tissue_description": "brain cortex"})
    print(result)

# 运行
asyncio.run(main())
```

## 注意事项

1. Qwen模型不支持flex服务层级，必须设置service_tier="default"
2. 对于thinking/no-thinking模式，可以通过reasoning_effort参数或自定义请求头控制
3. 本地部署的Qwen模型API地址为http://10.28.1.21:30080/v1
4. API不需要真实的key，但需要提供一个值（如"dummy-key"）