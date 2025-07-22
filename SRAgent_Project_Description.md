# SRAgent 项目开发级别描述文档

## 1. 项目概述

SRAgent 是一个基于多智能体（Multi-Agent）架构的工具，旨在简化与 NCBI Sequence Read Archive (SRA) 数据库及其他 Entrez 数据库的交互。它利用大型语言模型（LLM）的能力，通过自然语言指令执行复杂的生物信息学任务，例如查找数据集、提取元数据、处理Entrez ID等。项目设计注重模块化和可扩展性，使得开发者和AI模型能够理解其内部逻辑并进行定制化修改。

## 2. 项目结构

`SRAgent` 目录是项目的核心，其内部结构如下：

```
SRAgent/
├── SRAgent/                  # 核心Python包
│   ├── agents/               # 定义各种智能体（Agent）及其逻辑
│   ├── cli/                  # 命令行接口（CLI）相关代码
│   ├── db/                   # 数据库连接和操作相关代码
│   ├── workflows/            # 定义不同任务的工作流（Graph）
│   ├── tools/                # 智能体使用的工具函数
│   ├── __init__.py           # 包初始化文件
│   ├── check_srx_metadata.py # 检查SRX元数据脚本
│   ├── config.py             # 项目配置加载
│   ├── organisms.py          # 生物枚举定义
│   ├── search.py             # 搜索相关功能
│   ├── settings.yml          # 外部配置，包含模型、数据库等设置
│   └── utils.py              # 通用工具函数
├── assets/                   # 存放项目相关资产，如图片
├── notebooks/                # Jupyter Notebooks示例
├── scripts/                  # 辅助脚本
├── tests/                    # 单元测试和集成测试
├── .github/                  # GitHub Actions工作流配置
├── .gitignore                # Git忽略文件配置
├── DEPLOY.md                 # 部署文档
├── Dockerfile                # Dockerfile
├── LICENSE                   # 许可证文件
├── README.md                 # 项目主README
├── README_dataset_filter.md  # 数据集过滤相关README
├── moonshot_api.txt          # API相关信息
├── pyproject.toml            # Poetry项目配置
├── qwen_integration_guide.md # Qwen模型集成指南
└── ...
```

### 主要目录和文件说明：

*   **`SRAgent/SRAgent/agents/`**: 包含定义不同智能体行为的Python模块，例如 `find_datasets.py` 定义了查找数据集的智能体逻辑，`utils.py` 提供了模型设置和重试机制等通用功能。
*   **`SRAgent/SRAgent/cli/`**: 实现了命令行工具的各个子命令。`__main__.py` 是CLI的入口点，负责解析主命令和子命令，并分派到相应的处理函数。`find_datasets.py` 包含了 `find-datasets` 子命令的参数定义和主逻辑。
*   **`SRAgent/SRAgent/db/`**: 封装了与PostgreSQL数据库交互的逻辑，包括连接、创建表、插入、更新和查询数据等。
*   **`SRAgent/SRAgent/workflows/`**: 定义了基于 LangGraph 构建的复杂任务工作流。每个工作流（如 `find_datasets.py`）由一系列节点（智能体或工具）和边组成，描述了任务的执行流程。
*   **`SRAgent/SRAgent/tools/`**: 存放智能体在执行任务时可能调用的外部工具或内部辅助函数，例如与NCBI Entrez API交互的工具。
*   **`SRAgent/SRAgent/settings.yml`**: 项目的核心配置文件，以 YAML 格式存储。它定义了不同环境（`prod`, `test`, `default`）下的数据库连接信息、LLM模型名称、温度、推理努力（reasoning effort）、最大 token 数和服务层级等参数。这是调整项目行为的关键文件。
*   **`SRAgent/SRAgent/config.py`**: Python 模块，负责从 `settings.yml` 加载配置，并将其转换为 Python 对象供项目其他部分使用。它还处理环境变量的读取，例如 `ENTREZ_EMAIL` 和 `ENTREZ_API_KEY`。
*   **`SRAgent/SRAgent/utils.py`**: 包含项目范围内的通用实用函数，例如 `load_settings` 用于加载配置，以及 `set_model` 用于根据配置创建和配置LLM模型实例。

## 3. 核心组件详解

### 3.1 配置管理 (`settings.yml` 和 `config.py`)

项目的配置集中在 `settings.yml` 文件中，并通过 `config.py` 加载。`Dynaconf` 库用于管理这些配置，支持多环境配置和环境变量覆盖。

*   **`settings.yml`**: 
    *   **路径**: <mcfile name="settings.yml" path="/ssd2/xuyuan/SRAgent/SRAgent/settings.yml"></mcfile>
    *   **作用**: 定义了项目的各种可配置参数，包括：
        *   `moonshot`: 包含 `db_name` 以及不同智能体（如 `default`, `sragent`, `find_datasets` 等）的模型名称 (`models`)、温度 (`temperature`)、推理努力 (`reasoning_effort`)、最大 token 数 (`max_tokens`) 和服务层级 (`service_tier`)。
        *   `default`: 默认的数据库连接参数 (`db_host`, `db_port`, `db_user`, `db_password`, `db_timeout`)、服务层级 (`service_tier`)、弹性超时 (`flex_timeout`) 以及Qwen API的地址和密钥。
        *   `test`: 测试环境下的特定配置，通常会覆盖 `default` 中的一些参数。
    *   **修改方法**: 直接编辑 `settings.yml` 文件。例如，要修改 `prod` 环境下 `find_datasets` 智能体的 `max_tokens`，可以找到 `prod` 部分，然后修改 `moonshot.max_tokens.find_datasets` 的值。

```yaml:/ssd2/xuyuan/SRAgent/SRAgent/settings.yml
moonshot:
  db_name: "scagent"
  models:
    default: "Qwen3-235B-A22B"
    sragent: "Qwen3-235B-A22B"
    find_datasets: "Qwen3-235B-A22B" # 示例：为find_datasets指定模型
  temperature:
    default: 0.1
    sragent: 0.1
  reasoning_effort:
    default: "medium"
    sragent: "medium"
  max_tokens:
    default: 4096
    find_datasets: 4096 # 示例：为find_datasets指定最大token数
  service_tier:
    default: "default"
```

*   **`config.py`**: 
    *   **路径**: <mcfile name="config.py" path="/ssd2/xuyuan/SRAgent/SRAgent/config.py"></mcfile>
    *   **作用**: `Config` 类负责从 `settings.yml` 加载当前环境的配置，并将其属性化。它还从环境变量中读取敏感信息，如 `ENTREZ_EMAIL` 和 `ENTREZ_API_KEY`。
    *   **环境变量 `DYNACONF_ENV`**: 
        *   **作用**: `DYNACONF_ENV` 环境变量用于指定 `Dynaconf` 加载哪个环境的配置（例如 `prod` 或 `test`）。如果在运行命令时未显式设置，它将默认为 `prod`。
        *   **调用**: 在 `SRAgent/SRAgent/cli/__main__.py` 中，`load_dotenv(override=True)` 会加载 `.env` 文件中的环境变量，而 `os.environ["DYNACONF_ENV"] = args.tenant` 则允许通过命令行参数 `--tenant` 动态设置环境。
        *   **修改方法**: 
            1.  **通过命令行**: 在运行 `SRAgent` 命令时，使用 `--tenant` 参数指定环境，例如 `SRAgent find-datasets --tenant test ...`。
            2.  **通过环境变量**: 在执行 `SRAgent` 命令前，设置 `DYNACONF_ENV` 环境变量，例如 `export DYNACONF_ENV=test && SRAgent find-datasets ...`。
            3.  **通过 `.env` 文件**: 在项目根目录创建 `.env` 文件，并在其中添加 `DYNACONF_ENV=test`。`load_dotenv` 会自动加载此文件。

### 3.2 模型设置 (`SRAgent/SRAgent/agents/utils.py`)

`set_model` 函数是配置和实例化LLM模型的统一入口。它根据 `settings.yml` 中的配置或传入的参数来创建模型实例。

*   **`set_model` 函数**: 
    *   **路径**: <mcsymbol name="set_model" filename="utils.py" path="/ssd2/xuyuan/SRAgent/SRAgent/agents/utils.py" startline="150" type="function"></mcsymbol>
    *   **参数**: 
        *   `model_name` (str, optional): 模型名称，如 "Qwen3-235B-A22B"。如果未提供，则从 `settings.yml` 中获取。
        *   `temperature` (float, optional): 模型生成响应的随机性。较高的值表示更随机的输出。如果未提供，则从 `settings.yml` 中获取。
        *   `reasoning_effort` (str, optional): 推理努力级别，影响模型的思考模式和复杂性。可选值包括 "none", "low", "medium", "high", "max"。它会动态调整 `temperature` 和 `max_tokens`。如果未提供，则从 `settings.yml` 中获取。
        *   `agent_name` (str): 智能体的名称，用于从 `settings.yml` 中获取特定智能体的配置（例如 "find_datasets"）。
        *   `max_tokens` (int, optional): 模型生成响应的最大 token 数。如果未提供，则从 `settings.yml` 中获取。
        *   `service_tier` (str, optional): 服务层级，如 "default" 或 "flex"。影响模型的可用性和重试策略。如果未提供，则从 `settings.yml` 中获取。
    *   **修改方法**: 
        1.  **通过 `settings.yml`**: 这是推荐的方式。在 `settings.yml` 中为特定的 `agent_name` 或 `default` 配置 `model_name`, `temperature`, `reasoning_effort`, `max_tokens`, `service_tier`。例如，要调整 `find_datasets` 的 `reasoning_effort`，修改 `moonshot.reasoning_effort.find_datasets`。
        2.  **通过代码调用**: 在调用 `set_model` 的地方直接传入参数。例如，在 `create_get_entrez_ids_node` 函数中，`model = set_model(agent_name="get_entrez_ids")` 会使用 `settings.yml` 中 `get_entrez_ids` 的配置。如果需要硬编码覆盖，可以直接传入 `set_model(agent_name="get_entrez_ids", temperature=0.5)`。

### 3.3 命令行接口 (CLI) (`SRAgent/SRAgent/cli/`)

`SRAgent` 提供了一个强大的命令行接口，允许用户通过子命令与不同的功能进行交互。

*   **`SRAgent/SRAgent/cli/__main__.py`**: 
    *   **路径**: <mcfile name="__main__.py" path="/ssd2/xuyuan/SRAgent/SRAgent/cli/__main__.py"></mcfile>
    *   **作用**: 这是 `SRAgent` CLI 的主入口点。它使用 `argparse` 定义了主解析器和子命令解析器，并根据用户输入的子命令分派到相应的 `_main` 函数。
    *   **关键函数**: 
        *   <mcsymbol name="arg_parse" filename="__main__.py" path="/ssd2/xuyuan/SRAgent/SRAgent/cli/__main__.py" startline="28" type="function"></mcsymbol>: 定义了所有命令行参数，包括全局参数（如 `--no-progress`, `--no-summaries`）和各个子命令的参数。
        *   <mcsymbol name="main" filename="__main__.py" path="/ssd2/xuyuan/SRAgent/SRAgent/cli/__main__.py" startline="70" type="function"></mcsymbol>: 主执行函数，负责加载环境变量、连接数据库、创建表，并根据 `args.command` 调用相应的子命令主函数。
    *   **修改方法**: 
        *   要添加新的全局参数，修改 `arg_parse` 函数。
        *   要添加新的子命令，需要在 `arg_parse` 中添加新的 `subparsers`，并在 `main` 函数中添加相应的 `elif` 分支来调用新子命令的 `_main` 函数。

*   **`SRAgent/SRAgent/cli/find_datasets.py`**: 
    *   **路径**: <mcfile name="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/cli/find_datasets.py"></mcfile>
    *   **作用**: 定义了 `find-datasets` 子命令的所有参数和其主逻辑。
    *   **关键函数**: 
        *   <mcsymbol name="find_datasets_parser" filename="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/cli/find_datasets.py" startline="33" type="function"></mcsymbol>: 定义了 `find-datasets` 子命令特有的参数。
        *   <mcsymbol name="_find_datasets_main" filename="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/cli/find_datasets.py" startline="104" type="function"></mcsymbol>: `find-datasets` 子命令的异步主逻辑，负责设置环境、处理参数、创建工作流图并执行。
        *   <mcsymbol name="find_datasets_main" filename="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/cli/find_datasets.py" startline="192" type="function"></mcsymbol>: `find-datasets` 子命令的同步入口，调用异步主函数。
    *   **参数及其修改方法**: 
        *   `message` (str): 智能体的指令消息。直接在命令行中作为位置参数提供。
        *   `--max-datasets` (int, default=5): 要查找和处理的最大数据集数量。修改 `default` 值或在命令行中指定。
        *   `--min-date` (str, default=10年前): 搜索数据集的最早日期。修改 `default` 值或在命令行中指定，格式为 "YYYY/MM/DD"。
        *   `--max-date` (str, default=当前日期): 搜索数据集的最新日期。修改 `default` 值或在命令行中指定，格式为 "YYYY/MM/DD"。
        *   `--max-concurrency` (int, default=6): 最大并发进程数。修改 `default` 值或在命令行中指定。
        *   `--recursion-limit` (int, default=200): 最大递归限制。修改 `default` 值或在命令行中指定。
        *   `--organisms` (str, nargs='+', default=["human", "mouse"]): 要搜索的生物。修改 `default` 值或在命令行中指定，支持 "human-mouse" 或 "other-orgs"。
        *   `--use-database` (action='store_true', default=False): 是否使用数据库筛选现有数据集。在命令行中添加 `--use-database` 启用。
        *   `--tenant` (str, default='prod'): SRAgent SQL数据库的租户名称。修改 `default` 值或在命令行中指定，可选 `prod` 或 `test`。
        *   `--reprocess-existing` (action='store_true', default=False): 是否重新处理数据库中已存在的Entrez ID。在命令行中添加 `--reprocess-existing` 启用。
        *   `--write-graph` (str, metavar='FILE', default=None): 将工作流图写入文件并退出。在命令行中指定文件名和格式，如 `--write-graph workflow.png`。
        *   `--no-summaries` (action='store_true', default=False): 禁用步骤摘要输出。在命令行中添加 `--no-summaries` 启用。
        *   `--output-json` (str, default=None): 将结果输出为JSON文件。在命令行中指定文件名，如 `--output-json results.json`。

### 3.4 工作流 (`SRAgent/SRAgent/workflows/find_datasets.py`)

工作流是 `SRAgent` 组织复杂任务的核心。它们使用 LangGraph 库构建，通过定义一系列节点和它们之间的边来描述任务的执行流程。

*   **`SRAgent/SRAgent/workflows/find_datasets.py`**: 
    *   **路径**: <mcfile name="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/workflows/find_datasets.py"></mcfile>
    *   **作用**: 定义了 `find-datasets` 任务的工作流图。
    *   **关键函数**: 
        *   <mcsymbol name="create_find_datasets_graph" filename="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/workflows/find_datasets.py" startline="186" type="function"></mcsymbol>: 构建并返回 `find-datasets` 工作流的 LangGraph 图。这是理解任务执行流程的关键。
        *   **节点 (Nodes)**: 
            *   <mcsymbol name="create_find_datasets_node" filename="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/workflows/find_datasets.py" startline="34" type="function"></mcsymbol>: 调用 `find_datasets` 智能体来获取数据集。
            *   <mcsymbol name="create_get_entrez_ids_node" filename="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/workflows/find_datasets.py" startline="50" type="function"></mcsymbol>: 从消息中提取 Entrez ID 和数据库名称，并进行数据库筛选和数量限制。
            *   `srx_info_node` (通过 `create_SRX_info_graph()` 创建): 处理每个 SRX accession 的信息。
            *   <mcsymbol name="final_state" filename="find_datasets.py" path="/ssd2/xuyuan/SRAgent/SRAgent/workflows/find_datasets.py" startline="158" type="function"></mcsymbol>: 处理工作流的最终状态，收集结果。
        *   **边 (Edges)**: 定义了节点之间的流转逻辑，包括条件流转（如 `continue_to_srx_info`）。
    *   **修改方法**: 
        *   要修改工作流的逻辑或添加新步骤，可以在 `create_find_datasets_graph` 中添加新的节点或修改现有节点的逻辑。
        *   要调整节点之间的流转，修改 `workflow.add_edge` 或 `workflow.add_conditional_edges` 的定义。
        *   例如，如果想在提取 Entrez ID 后增加一个验证步骤，可以添加一个新的验证节点，并修改 `get_entrez_ids_node` 到 `srx_info_node` 的边，使其先经过验证节点。

### 3.5 数据库交互 (`SRAgent/SRAgent/db/`)

`SRAgent` 使用 PostgreSQL 数据库来存储和管理数据集元数据。`SRAgent/SRAgent/db/` 目录包含了所有数据库操作的抽象。

*   **目录**: <mcfolder name="db/" path="/ssd2/xuyuan/SRAgent/SRAgent/db/"></mcfolder>
*   **关键文件和函数**: 
    *   `connect.py`: 包含 <mcsymbol name="db_connect" filename="connect.py" path="/ssd2/xuyuan/SRAgent/SRAgent/db/connect.py" startline="10" type="function"></mcsymbol> 函数，用于建立数据库连接。数据库连接参数从 `config.py` 加载。
    *   `create.py`: 包含 <mcsymbol name="create_srx_metadata" filename="create.py" path="/ssd2/xuyuan/SRAgent/SRAgent/db/create.py" startline="10" type="function"></mcsymbol> 和 <mcsymbol name="create_srx_srr" filename="create.py" path="/ssd2/xuyuan/SRAgent/SRAgent/db/create.py" startline="29" type="function"></mcsymbol> 函数，用于创建数据库表。
    *   `upsert.py`: 包含 <mcsymbol name="db_upsert" filename="upsert.py" path="/ssd2/xuyuan/SRAgent/SRAgent/db/upsert.py" startline="10" type="function"></mcsymbol> 函数，用于插入或更新数据。
    *   `get.py`: 包含 <mcsymbol name="db_get_entrez_ids" filename="get.py" path="/ssd2/xuyuan/SRAgent/SRAgent/db/get.py" startline="10" type="function"></mcsymbol> 等函数，用于从数据库中查询数据。
*   **修改方法**: 
    *   要修改数据库连接参数，请编辑 `settings.yml` 中的 `default` 或特定环境下的 `db_host`, `db_port`, `db_user`, `db_password`, `db_timeout`。
    *   要修改数据库模式或添加新表，需要修改 `create.py` 中的相应函数。
    *   要修改数据操作逻辑（插入、更新、查询），修改 `upsert.py` 和 `get.py` 中的函数。

## 4. 开发与扩展

### 4.1 添加新功能或智能体

1.  **定义智能体逻辑**: 在 `SRAgent/SRAgent/agents/` 目录下创建新的Python模块，定义新智能体的核心逻辑和功能。
2.  **创建命令行接口**: 如果新功能需要通过CLI访问，在 `SRAgent/SRAgent/cli/` 目录下创建新的模块，定义其参数解析器和主函数，并在 `SRAgent/SRAgent/cli/__main__.py` 中注册。
3.  **构建工作流**: 如果新功能涉及复杂的多步骤任务，在 `SRAgent/SRAgent/workflows/` 目录下创建新的模块，使用 LangGraph 构建工作流图，将新智能体作为节点集成进去。
4.  **配置模型**: 在 `settings.yml` 中为新智能体添加模型配置（`model_name`, `temperature`, `reasoning_effort`, `max_tokens`, `service_tier`）。
5.  **添加工具**: 如果新智能体需要与外部API或特定数据源交互，在 `SRAgent/SRAgent/tools/` 目录下添加相应的工具函数。

### 4.2 调整现有功能

*   **调整模型行为**: 
    *   修改 `settings.yml` 中对应智能体的 `temperature` 和 `reasoning_effort` 来调整模型的创造性和思考深度。
    *   修改 `max_tokens` 来控制模型的输入输出长度。
    *   修改 `model_name` 来切换不同的LLM模型。
*   **修改工作流流程**: 
    *   编辑 `SRAgent/SRAgent/workflows/` 目录下相应的工作流文件，修改节点之间的边或添加/删除节点。
    *   例如，在 `find_datasets` 工作流中，如果想在获取 Entrez ID 后增加一个人工审核步骤，可以在 `create_find_datasets_graph` 中添加一个新节点，并调整边。
*   **修改命令行参数**: 
    *   编辑 `SRAgent/SRAgent/cli/` 目录下相应子命令的 `_parser` 函数，添加、修改或删除命令行参数。
    *   在 `_main` 函数中，根据新的参数调整逻辑。

### 4.3 数据库模式变更

*   如果需要修改数据库表结构或添加新表，请修改 `SRAgent/SRAgent/db/create.py` 中的相应函数。
*   确保在 `SRAgent/SRAgent/cli/__main__.py` 的 `main` 函数中调用了新的表创建函数，以便在项目启动时自动创建表。
*   如果数据模型发生变化，可能需要更新 `SRAgent/SRAgent/db/upsert.py` 和 `SRAgent/SRAgent/db/get.py` 中的数据操作逻辑。

## 5. 总结

`SRAgent` 项目通过模块化的设计、清晰的配置管理和基于 LangGraph 的工作流，提供了一个灵活且强大的生物信息学工具。理解 `settings.yml`、`config.py`、`agents/utils.py` 中的模型设置、`cli/` 中的命令行接口以及 `workflows/` 中的工作流构建是进行开发和扩展的关键。通过遵循本文档的指导，用户和AI模型可以有效地理解、修改和扩展 `SRAgent` 的功能，以适应不断变化的需求。