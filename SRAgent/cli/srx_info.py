# import
# 导入部分

# 标准库导入
import os
import sys
import asyncio
import argparse
import pandas as pd
from typing import List, Optional, Callable

# 第三方库导入
from Bio import Entrez

# 项目内部模块导入
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.srx_info import create_SRX_info_graph
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_step_summary_chain
from SRAgent.db.connect import db_connect 
from SRAgent.db.get import db_get_srx_records
from SRAgent.tools.utils import set_entrez_access

# 函数定义
def SRX_info_agent_parser(subparsers):
    """
    配置 SRX 信息代理的命令行参数解析器。

    Args:
        subparsers: argparse 的子解析器对象，用于添加新的子命令。
    """
    help_msg = 'SRX_info 代理: 获取 SRA 实验的元数据。'
    description = """
    该代理用于查询 NCBI SRA 数据库，获取指定 Entrez ID 的实验元数据。
    支持从命令行直接输入 Entrez ID 或从 CSV 文件读取。
    可以将结果存储到 SRAgent 的 SQL 数据库中。
    """
    sub_parser = subparsers.add_parser(
        'srx-info', help=help_msg, description=description, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=SRX_info_agent_main)
    sub_parser.add_argument(
        'entrez_ids', type=str, nargs='+', 
        help='要查询的 Entrez ID 列表（或包含“entrez_id”列的 CSV 文件）'
    )    
    sub_parser.add_argument(
        '--database', type=str, default='sra', choices=['gds', 'sra'], 
        help='Entrez ID 的来源数据库'
    )
    sub_parser.add_argument(
        '--max-concurrency', type=int, default=6, 
        help='最大并发处理进程数'
    )
    sub_parser.add_argument(
        '--recursion-limit', type=int, default=200, 
        help='最大递归限制'
    )
    sub_parser.add_argument(
        '--max-parallel', type=int, default=3, 
        help='Entrez ID 的最大并行处理数'
    )
    sub_parser.add_argument(
        '--use-database', action='store_true', default=False, 
        help='将结果添加到 SRAgent SQL 数据库'
    )
    sub_parser.add_argument(
        '--tenant', type=str, default='prod',
        choices=['prod', 'test'],
        help='SRAgent SQL 数据库的租户名称'
    )
    sub_parser.add_argument(
        '--reprocess-existing', action='store_true', default=False, 
        help='重新处理 SRAgent 数据库中已存在的 Entrez ID，而不是忽略它们（假设已启用 --use-database）'
    )
    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='将工作流图写入文件并退出（支持 .png, .svg, .pdf, .mermaid 格式）'
    )

async def _process_single_entrez_id(
    entrez_id: str, 
    database: str, 
    graph: Callable, 
    step_summary_chain: Optional[Callable], 
    config: dict, 
    no_summaries: bool
):
    """
    处理单个 Entrez ID 的异步函数。

    Args:
        entrez_id (str): 要处理的 Entrez ID。
        database (str): 要使用的数据库。
        graph (Callable): 要执行的工作流图。
        step_summary_chain (Optional[Callable]): 用于生成步骤摘要的链（如果提供）。
        config (dict): 代理的配置字典。
        no_summaries (bool): 是否不打印步骤摘要。
    """
    # 准备输入数据
    input_data = {
        "entrez_id": entrez_id, 
        "database": database,
    }
    final_state = None
    step_count = 0
    
    # 流式处理图的每一步
    async for step in graph.astream(input_data, config=config):
        step_count += 1
        final_state = step
        
        # 打印步骤摘要或节点信息
        if step_summary_chain:
            msg = await step_summary_chain.ainvoke({"step": step})
            print(f"[{entrez_id}] 步骤 {step_count}: {msg.content}")
        else:
            nodes = ",".join(list(step.keys()))
            print(f"[{entrez_id}] 步骤 {step_count}: {nodes}")

    # 显示最终结果
    if final_state:
        print(f"#-- Entrez ID {entrez_id} 的最终结果 --#")
        try:
            # 尝试从最终状态节点中提取并打印消息内容
            print(final_state["final_state_node"]["messages"][-1].content)
        except KeyError:
            # 如果没有找到消息内容，则表示处理被跳过
            print("处理已跳过")
    print("#---------------------------------------------#")

async def _SRX_info_agent_main(args):
    """
    调用 SRX 信息代理的主函数。

    Args:
        args: 命令行解析后的参数对象。
    """
    # 根据参数过滤 Entrez ID
    if args.use_database and not args.reprocess_existing:
        existing_ids = set()
        with db_connect() as conn:
            # 从数据库获取已存在的 SRX 记录
            existing_ids = set(db_get_srx_records(conn))
        # 过滤掉已存在的 Entrez ID
        args.entrez_ids = [x for x in args.entrez_ids if x not in existing_ids]
        if len(args.entrez_ids) == 0:
            print("所有 Entrez ID 都已存在于元数据数据库中。", file=sys.stderr)
            return 0

    # 设置 Entrez 邮箱和 API 密钥
    set_entrez_access()

    # 创建监督代理的工作流图
    graph = create_SRX_info_graph()
    # 根据是否需要摘要创建步骤摘要链
    if not args.no_summaries:
        step_summary_chain = create_step_summary_chain()
    else:
        step_summary_chain = None

    # 配置代理的调用参数
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit,
        "configurable": {
            "use_database": args.use_database,
            "reprocess_existing": args.reprocess_existing,
        }
    }

    # 创建信号量以限制并发处理数量
    semaphore = asyncio.Semaphore(args.max_parallel)

    # 定义一个异步函数，用于在信号量控制下处理单个 Entrez ID
    async def _process_with_semaphore(entrez_id):
        async with semaphore:
            await _process_single_entrez_id(
                entrez_id,
                args.database,
                graph,
                step_summary_chain,
                config,
                args.no_summaries,
            )

    # 为每个 Entrez ID 创建处理任务
    tasks = [_process_with_semaphore(entrez_id) for entrez_id in args.entrez_ids]
    
    # 并发运行所有任务，并限制并发数量
    await asyncio.gather(*tasks)

def SRX_info_agent_main(args):
    """
    SRX 信息代理的主入口函数。

    Args:
        args: 命令行解析后的参数对象。
    """
    # 设置租户环境变量
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant
    
    # 处理写入图选项
    if args.write_graph:
        handle_write_graph_option(create_SRX_info_graph, args.write_graph)
        return

    # 如果 entrez_ids 是一个 CSV 文件，则读取其中的 Entrez ID
    if args.entrez_ids[0].endswith(".csv") and os.path.exists(args.entrez_ids[0]):
        df = pd.read_csv(args.entrez_ids[0])
        if "entrez_id" not in df.columns:
            print("CSV 文件中未找到 'entrez_id' 列", file=sys.stderr)
            return 1
        args.entrez_ids = df["entrez_id"].unique().astype(str).tolist()
        
    # 过滤非数字的 Entrez ID
    problem_entrez_ids = [x for x in args.entrez_ids if not x.isnumeric()]
    if problem_entrez_ids:
        print("发现无效的 Entrez ID:", file=sys.stderr)
        for x in problem_entrez_ids:
            print(x, file=sys.stderr)
        return 1
    
    # 运行代理
    asyncio.run(_SRX_info_agent_main(args))

# 主程序入口
if __name__ == '__main__':
    pass