# import
# 导入必要的库
import os
import asyncio
from Bio import Entrez # 用于与NCBI Entrez数据库交互
from langchain_core.messages import HumanMessage # 用于创建人类消息
from SRAgent.cli.utils import CustomFormatter # 自定义格式化器，用于美化命令行输出
from SRAgent.agents.entrez import create_entrez_agent # 用于创建Entrez智能体
from SRAgent.workflows.graph_utils import handle_write_graph_option # 处理写入工作流图的选项
from SRAgent.agents.display import create_agent_stream, display_final_results # 用于创建智能体流和显示最终结果
from SRAgent.tools.utils import set_entrez_access # 设置Entrez访问权限

# 函数定义
def entrez_agent_parser(subparsers):
    """
    为Entrez智能体添加命令行参数解析器。
    """
    help_msg = 'Entrez智能体: 用于与Entrez数据库交互的通用智能体。'
    description_msg = """
    # 示例提示:
    1. "将GSE121737转换为SRX登录号"
    2. "获取GSE196830的任何可用出版物"
    3. "获取SRX4967527的SRR登录号"
    4. "SRR8147022是双端Illumina数据吗?"
    5. "SRP309720是10X Genomics数据吗?"
    """
    sub_parser = subparsers.add_parser(
        'entrez', 
        help=help_msg, 
        description=description_msg, 
        formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=entrez_agent_main) # 设置默认执行函数
    sub_parser.add_argument('prompt', type=str, help='给智能体的提示')    
    sub_parser.add_argument('--max-concurrency', type=int, default=3, 
                            help='最大并发进程数')
    sub_parser.add_argument('--recursion-limit', type=int, default=40,
                            help='最大递归限制')
    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='将工作流图写入文件并退出（支持.png, .svg, .pdf, .mermaid格式）'
    )
    
def entrez_agent_main(args):
    """
    调用Entrez智能体的主函数。
    """
    # 设置Entrez的邮箱和API密钥
    set_entrez_access()
    
    # 处理写入工作流图的选项
    if args.write_graph:
        handle_write_graph_option(create_entrez_agent, args.write_graph)
        return

    # 以流式方式调用智能体
    config = {
        "max_concurrency" : args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input_messages = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input_messages, create_entrez_agent, config, 
            summarize_steps=not args.no_summaries,
            no_progress=args.no_progress
        )
    )
    
    # 显示最终结果并进行富文本格式化
    display_final_results(results)

# 主程序入口
if __name__ == '__main__':
    pass