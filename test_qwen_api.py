import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from modified_utils import set_model

# 加载环境变量
load_dotenv(override=True)

# 测试函数1：直接使用requests库调用Qwen API（thinking模式）
def test_qwen_api_thinking():
    print("\n===== 测试Qwen API (thinking模式) =====\n")
    
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
        "max_tokens": 1000  # 减小token数以加快响应
    }
    
    print("请求数据:")
    print(json.dumps(data, indent=2))
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        print("\n响应数据:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 提取并打印生成的内容
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print("\n生成的内容:")
            print(content)
    except Exception as e:
        print(f"请求失败: {e}")

# 测试函数2：直接使用requests库调用Qwen API（no-thinking模式）
def test_qwen_api_no_thinking():
    print("\n===== 测试Qwen API (no-thinking模式) =====\n")
    
    url = "http://10.28.1.21:30080/v1/chat/completions"
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
        "max_tokens": 1000,  # 减小token数以加快响应
        "presence_penalty": 1.5
    }
    
    print("请求数据:")
    print(json.dumps(data, indent=2))
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        print("\n响应数据:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 提取并打印生成的内容
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print("\n生成的内容:")
            print(content)
    except Exception as e:
        print(f"请求失败: {e}")

# 测试函数3：使用LangChain集成Qwen API
def test_langchain_qwen():
    print("\n===== 测试LangChain集成Qwen API =====\n")
    
    try:
        # 使用修改后的set_model函数创建模型
        model = set_model(model_name="Qwen3-235B-A22B", temperature=0.7, reasoning_effort=None, max_tokens=1000)
        print(f"模型信息: {model}")
        
        # 创建消息
        message = HumanMessage(content="Give me a short introduction to large language models.")
        
        # 调用模型
        print("\n发送请求...")
        response = model.invoke([message])
        
        print("\n响应结果:")
        print(response)
        print("\n响应内容:")
        print(response.content)
    except Exception as e:
        print(f"LangChain调用失败: {e}")

# 主函数
if __name__ == "__main__":
    # 测试直接API调用（thinking模式）
    test_qwen_api_thinking()
    
    # 测试直接API调用（no-thinking模式）
    test_qwen_api_no_thinking()
    
    # 测试LangChain集成
    test_langchain_qwen()