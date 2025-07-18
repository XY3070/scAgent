# import
# 导入必要的库

# 标准库
import os
import sys
import asyncio
import argparse
from typing import List
from datetime import datetime, timedelta

# 第三方库
from Bio import Entrez # 用于与NCBI Entrez数据库交互
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # 用于处理LangChain消息

# 项目内部模块
from SRAgent.cli.utils import CustomFormatter # 自定义格式化器，用于美化命令行输出
from SRAgent.workflows.find_datasets import create_find_datasets_graph # 用于创建查找数据集的工作流图
from SRAgent.workflows.graph_utils import handle_write_graph_option # 处理写入工作流图的选项
from SRAgent.agents.display import create_step_summary_chain # 用于创建步骤摘要链
from SRAgent.tools.utils import set_entrez_access # 设置Entrez访问权限
from SRAgent.organisms import OrganismEnum # 生物枚举类型

# 生物参数
human_mouse = [OrganismEnum.HUMAN.name.lower(), OrganismEnum.MOUSE.name.lower()] # 人类和小鼠的名称列表
other_orgs = [
    org.name.lower().replace("_", " ") for org in OrganismEnum 
    if org not in [OrganismEnum.HUMAN, OrganismEnum.MOUSE]
] # 其他生物的名称列表

# 函数定义
def find_datasets_parser(subparsers):
    """
    为查找数据集功能添加命令行参数解析器。
    """
    help_msg = '获取数据集并通过srx-info工作流处理每个数据集'
    description_msg = """
    示例消息: "在SRA数据库中获取最近的单细胞RNA-seq数据集"
    """
    sub_parser = subparsers.add_parser(
        'find-datasets', 
        help=help_msg, 
        description=description_msg, 
        formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=find_datasets_main) # 设置默认执行函数
    sub_parser.add_argument(
        'message', type=str, help='指示智能体的消息。参见描述'
    ) 
    sub_parser.add_argument(
        '--max-datasets', type=int, default=5, help='要查找和处理的最大数据集数量'
    )
    sub_parser.add_argument(
        '--min-date', type=str, 
        default=(datetime.now() - timedelta(days=365 * 10)).strftime("%Y/%m/%d"), 
        help='搜索数据集的最早日期'
    )
    sub_parser.add_argument(
        '--max-date', type=str, default=datetime.now().strftime("%Y/%m/%d"),
        help='搜索数据集的最新日期'
    )
    sub_parser.add_argument(
        '--max-concurrency', type=int, default=6, help='最大并发进程数'
    )
    sub_parser.add_argument(
        '--recursion-limit', type=int, default=200, help='最大递归限制'
    )
    sub_parser.add_argument(
        '-o', '--organisms', type=str, nargs='+', default=human_mouse,
        choices=human_mouse + other_orgs + ["human-mouse", "other-orgs"],
        help='要搜索的生物。使用"human-mouse"选择人类/小鼠，或使用"other-orgs"选择所有其他生物'
    )
    sub_parser.add_argument(
        '--use-database', action='store_true', default=False, 
        help='使用scBaseCount数据库筛选现有数据集，以便添加新发现的数据集'
    )  
    sub_parser.add_argument(
        '--tenant', type=str, default='prod',
        choices=['prod', 'test'],
        help='SRAgent SQL数据库的租户名称'
    )
    sub_parser.add_argument(
        '--reprocess-existing', action='store_true', default=False, 
        help='重新处理scBaseCount数据库中已存在的Entrez ID，而不是忽略现有ID（假设使用--use-database）'
    )
    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='将工作流图写入文件并退出（支持.png, .svg, .pdf, .mermaid格式）'
    )  

async def _find_datasets_main(args):
    """
    调用查找数据集工作流的主函数。
    """
    # 设置租户
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant

    # 设置Entrez的邮箱和API密钥
    set_entrez_access()
    
    # 处理写入工作流图的选项
    if args.write_graph:
        handle_write_graph_option(create_find_datasets_graph, args.write_graph)
        return
    
    # 创建监督智能体
    graph = create_find_datasets_graph()
    
    # 如果不禁用摘要，则创建步骤摘要链
    if not args.no_summaries:
        step_summary_chain = create_step_summary_chain()

    # 格式化生物体参数
    if "human-mouse" in args.organisms:
        args.organisms.remove("human-mouse")
        args.organisms += human_mouse
    if "other-orgs" in args.organisms:
        args.organisms.remove("other-orgs")
        args.organisms += other_orgs
    args.organisms = sorted(list(set(args.organisms))) # 去重并排序

    # 设置图的配置
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit,
        "configurable": {
            "organisms": args.organisms,
            "max_datasets": args.max_datasets,
            "use_database": args.use_database,
            "reprocess_existing": args.reprocess_existing,
            "min_date": args.min_date,
            "max_date": args.max_date,
        }
    }
    input_messages = {"messages" : [HumanMessage(content=args.message)]}

    # 调用图并进行流式处理
    final_state = None
    i = 0
    async for step in graph.astream(input_messages, config=config):
        i += 1
        final_state = step
        if args.no_summaries:
            nodes = ",".join(list(step.keys()))
            print(f"步骤 {i}: {nodes}")
        else:
            msg = await step_summary_chain.ainvoke({"step": step})
            print(f"步骤 {i}: {msg.content}")

    # 打印最终结果
    if final_state:
        print(f"\n#-- 最终结果 --#")
        try:
            for msg in final_state["final_state_node"]["messages"]:
                try:
                    print(msg.content)
                except AttributeError:
                    if isinstance(msg, list):
                        for x in msg:
                            print(x)
                    else:
                        print(msg)
        except KeyError:
            print("处理已跳过")

def find_datasets_main(args):
    """
    查找数据集的主入口函数，运行异步主函数。
    """
    asyncio.run(_find_datasets_main(args))

# 主程序入口
if __name__ == '__main__':
    pass