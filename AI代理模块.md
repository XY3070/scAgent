# AI代理模块 (agents/)

## 概述

AI代理模块实现了基于语言模型的智能代理，用于处理和分析生物信息学数据。该模块包含多个专门的代理，用于执行不同的任务，如搜索、提取和处理数据。

## 文件结构

```
SRAgent/agents/
├── bigquery.py          # BigQuery代理
├── display.py           # 显示结果代理
├── efetch.py            # Entrez Fetch代理
├── elink.py             # Entrez Link代理
├── entrez.py            # Entrez主代理
├── entrez_convert.py    # Entrez转换代理
├── esearch.py           # Entrez Search代理
├── esummary.py          # Entrez Summary代理
├── ncbi_fetch.py        # NCBI数据获取代理
├── sequences.py         # 序列处理代理
└── utils.py             # 工具函数
```

## 主要组件功能

### utils.py

提供AI代理模块的通用工具函数，包括设置模型、加载配置等。

**主要函数**：
- `load_settings()`: 从settings.yml文件加载设置。
- `set_model()`: 设置语言模型，支持OpenAI、Anthropic和Qwen模型，并处理模型初始化时的超时设置。

### 模型配置

模型配置通过 `settings.yml` 文件进行管理，并通过 `set_model` 函数加载。该函数根据模型名称、温度、推理强度、最大 token 数和服务层级来创建模型实例。API 密钥和基础 URL 从 `settings.yml` 或环境变量中获取。

**关键点：**

*   **动态模型选择：** `set_model` 函数根据 `model_name` 动态选择 `FlexTierChatOpenAI` 或 `ChatAnthropic`。
*   **弹性层级支持：** `FlexTierChatOpenAI` 扩展了 `ChatOpenAI`，支持弹性层级和回退机制。
*   **推理强度：** 对于支持推理强度的模型，`reasoning_effort` 会覆盖 `temperature` 设置。
*   **API 密钥和基础 URL：** 优先从 `settings.yml` 获取 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 和 `Qwen` 模型的 `qwen_api_base`，如果 `settings.yml` 中未配置，则从环境变量中获取。
*   **超时设置：** 通过 `request_timeout` 参数设置模型调用的超时时间，支持根据服务层级动态调整。

### extract_meta.py

实现元数据提取代理，用于从实验数据中提取元数据。

**主要功能：**
- 使用异步并发处理提高处理效率
- 动态调整并发数量以优化性能
- 自动处理超时和重试机制


### entrez.py

实现Entrez主代理，用于协调其他Entrez相关代理的工作。

**主要函数**：
- `create_entrez_agent()`: 创建Entrez代理，集成esearch、esummary、efetch和elink代理。

### display.py

实现结果显示代理，用于格式化和显示代理的输出结果。

**主要函数**：
- `create_agent_stream()`: 创建代理流，用于实时显示代理的输出。
- `display_final_results()`: 显示最终结果。

## 与需求的关系

该模块部分支持用户需求中的使用本地Qwen AI agent进行数据处理的功能，但存在以下问题：

## 冗余和不符合需求的内容

**不符合需求的内容**：

1. **Entrez相关代理**：
   - `efetch.py`、`elink.py`、`entrez.py`、`entrez_convert.py`、`esearch.py`、`esummary.py`、`ncbi_fetch.py`
   - 这些文件实现了与Entrez在线数据库交互的代理，与用户需求中不使用Entrez访问在线数据库的要求不符。

2. **BigQuery代理**：
   - `bigquery.py`
   - 实现了与Google BigQuery服务交互的代理，可能不是本地数据处理所必需的。

## 增强型工作流与AI代理集成

增强型工作流整合了预过滤、分类和增强型AI优化元数据提取功能，为AI代理提供了更灵活的数据处理能力。

**主要功能**：
- 支持可选的过滤条件参数（测序策略、癌症状态、关键词搜索）。
- 生成多种输出格式，包括用于AI代理的完整增强导出JSON、用于快速AI处理的结构化元数据JSON以及AI处理指令Markdown文件。
- 提供灵活的分类选项，可根据项目类型对数据集进行分类。

## 并发处理和超时优化

为了提高AI代理的处理效率和稳定性，我们对并发处理和超时机制进行了优化：

1. **动态并发控制**：
   - 使用 `asyncio.Semaphore` 控制并发数量，避免资源过载
   - 根据模型响应时间动态调整并发数，优化处理效率
   - 设置最大和最小并发数限制，确保系统稳定性

2. **超时设置优化**：
   - 在 `settings.yml` 中增加了 `db_timeout` 和 `flex_timeout` 的值，以适应更复杂的处理需求
   - 在模型调用中显式设置超时参数，提高请求的可控性

3. **重试机制**：
   - 实现了更完善的重试机制，处理临时性错误和超时情况
   - 在重试之间增加延迟，避免连续请求导致的问题

## 改进建议

1. **移除或禁用Entrez相关代理**：
   - 将与Entrez相关的代理标记为禁用，或者完全移除，以符合不使用在线数据库的需求。

2. **添加Qwen AI模型支持**：
   - 在`utils.py`的`set_model()`函数中添加对Qwen模型的支持。✅ 已实现
   - 创建专门的Qwen代理类，用于本地数据处理。

3. **优化代理架构**：
   - 重构代理架构，使其更适合本地数据处理的需求。
   - 添加专门用于处理PostgreSQL数据的代理。
   - ✅ 已实现：通过增强型工作流为AI代理提供更灵活的数据处理能力，支持可选的过滤条件参数。

4. **增强文档和示例**：
   - 为每个代理添加详细的文档和使用示例。
   - 提供代理配置的最佳实践。
   - ✅ 已实现：为增强型工作流提供了详细的文档和使用示例。

5. **持续优化并发处理**：
   - 定期评估和调整并发处理参数，以适应不同的数据处理需求
   - 收集和分析处理日志，进一步优化性能