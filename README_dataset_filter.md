# 数据集筛选功能说明文档

## 功能概述

本项目实现了一个数据集筛选系统，可以根据用户指定的条件从数据库中检索单细胞RNA测序数据集。系统包含三个主要功能模块：

1. **组织本体查询（Tissue Ontology）**：将自由文本描述的组织名称映射到Uberon本体术语ID
2. **数据集筛选（Filter Datasets）**：根据多种条件筛选数据集
3. **集成搜索（Integrated Search）**：结合组织本体查询和数据集筛选功能，提供更强大的搜索能力

## 筛选条件

系统支持以下筛选条件：

### 必选条件

1. **物种（Organism）**：如homo sapiens（人类）
2. **排除细胞系（Exclude Cell Line）**：是否排除细胞系样本
3. **数据可获取（Data Available）**：确保数据可以获取，提供数据库来源和ID
4. **癌症标注（Cancer Annotation）**：必须标注是否为癌症患者组织或肿瘤组织
5. **测序方法（Sequencing Method）**：如10X genomics、smart-seq2等
6. **组织来源（Tissue Source）**：组织来源必须明确

### 可选条件

1. **发表文章（Publication）**：是否有对应发表文章，PMID或DOI
2. **样本量（Sample Size）**：是否有样本量信息
3. **国籍/地区（Nationality/Region）**：是否有国籍/地区信息

## 代码结构

### 核心文件

- `SRAgent/workflows/filter_datasets.py`：数据集筛选工作流
- `SRAgent/workflows/integrated_search.py`：集成搜索工作流
- `SRAgent/agents/filter_datasets.py`：数据集筛选代理

### 测试文件

- `test_filter_datasets_qwen.py`：测试数据集筛选功能
- `test_integrated_search_qwen.py`：测试集成搜索功能

## 使用方法

### 1. 数据集筛选

```python
import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from SRAgent.workflows.filter_datasets import create_filter_datasets_workflow

# 加载环境变量
load_dotenv()

# 设置数据库连接信息
os.environ["DB_HOST"] = "10.28.1.24"
os.environ["DB_PORT"] = "5432"
os.environ["DB_USER"] = "yanglab"
os.environ["GCP_SQL_DB_PASSWORD"] = "labyang"
os.environ["DB_NAME"] = "screcounter"

async def main():
    # 创建工作流
    workflow = create_filter_datasets_workflow(model_name="Qwen3-235B-A22B", service_tier="qwen")
    
    # 使用自然语言查询
    query = HumanMessage(content="Find datasets for homo sapiens with 10X genomics sequencing method")
    result = await workflow.ainvoke({"messages": [query]}, config={})
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 集成搜索

```python
import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from SRAgent.workflows.integrated_search import create_integrated_search_workflow

# 加载环境变量
load_dotenv()

# 设置数据库连接信息
os.environ["DB_HOST"] = "10.28.1.24"
os.environ["DB_PORT"] = "5432"
os.environ["DB_USER"] = "yanglab"
os.environ["GCP_SQL_DB_PASSWORD"] = "labyang"
os.environ["DB_NAME"] = "screcounter"

async def main():
    # 创建工作流
    workflow = create_integrated_search_workflow(model_name="Qwen3-235B-A22B", service_tier="qwen")
    
    # 集成查询（组织本体 + 数据集筛选）
    query = HumanMessage(content="Find brain cortex datasets for homo sapiens using 10X genomics")
    result = await workflow.ainvoke({"messages": [query]}, config={})
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## 查询示例

### 基本查询

- "Find datasets for homo sapiens with 10X genomics sequencing method"
- "Show me smart-seq2 data from mouse samples"
- "I need cancer datasets with publication information"

### 组织特定查询

- "What is the Uberon ID for brain cortex?"
- "Find brain cortex datasets for homo sapiens"

### 复杂查询

```
Find datasets with the following criteria:
1. Organism: homo sapiens
2. Tissue: liver
3. Exclude cell lines
4. Data must be available
5. Must have cancer annotation
6. Sequencing method: smart-seq2
7. Should have publication information
```

## 设计考虑

1. **低侵入性**：新功能与现有系统松耦合，不影响原有功能
2. **易扩展**：可以轻松添加新的筛选条件或搜索功能
3. **高效查询**：使用SQL优化查询，限制结果数量以提高性能
4. **用户友好**：支持自然语言查询，提供详细的结果展示
5. **错误处理**：包含全面的错误处理机制，确保系统稳定性

## 未来改进

1. 添加更多筛选条件，如基因表达水平、细胞类型等
2. 优化查询性能，添加索引和缓存机制
3. 提供更丰富的结果展示，如可视化和统计分析
4. 实现更智能的查询理解，支持更复杂的自然语言查询
5. 添加用户反馈机制，不断改进搜索结果的相关性