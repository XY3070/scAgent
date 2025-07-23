# import
# 导入标准库
import os
import asyncio

# 导入第三方库
from Bio import Entrez
from langchain_core.messages import HumanMessage

# 导入项目内部模块
from SRAgent.cli.utils import CustomFormatter
from SRAgent.agents.sragent import create_sragent_agent
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_agent_stream, display_final_results
from SRAgent.tools.utils import set_entrez_access

# 定义函数
def sragent_parser(subparsers):
    """
    为 SRAgent 设置命令行参数解析器。

    Args:
        subparsers: argparse 的子解析器对象，用于添加新的子命令。

    Returns:
        None
    """
    help_msg = 'SRAgent: 用于处理序列数据的高级代理。'
    description = """
    # 示例提示：
    1. "将 GSE121737 转换为 SRX 登录号"
    2. "SRX25994842 是 Illumina 序列数据、10X Genomics 数据，以及属于哪个生物体？"
    3. "列出 SRX20554853 数据集的合作者"
    4. "获取 SRX20554853 的所有 SRR 登录号"
    5. "SRP309720 是双端测序数据吗？"
    """
    sub_parser = subparsers.add_parser(
        'sragent', help=help_msg, description=description, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=sragent_main)
    sub_parser.add_argument('prompt', type=str, help='代理的提示。') 
    sub_parser.add_argument('--max-concurrency', type=int, default=3, 
                            help='最大并发进程数。')
    sub_parser.add_argument('--recursion-limit', type=int, default=40,
                            help='最大递归限制。')
    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='将工作流图写入文件并退出（支持 .png, .svg, .pdf, .mermaid 格式）。'
    )

def sragent_main(args):
    """
    调用 SRAgent 代理的主函数。

    Args:
        args: 命令行解析后的参数对象。

    Returns:
        None
    """
    # 设置 Entrez 邮箱和 API 密钥
    set_entrez_access()
    
    # 处理写入图选项
    if args.write_graph:
        handle_write_graph_option(create_sragent_agent, args.write_graph)
        return

    # 以流式方式调用代理
    config = {
        "max_concurrency" : args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input_data = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input_data, create_sragent_agent, config, 
            summarize_steps=not args.no_summaries,
            no_progress=args.no_progress
        )
    )
    
    # 以富文本格式显示最终结果
    display_final_results(results)

# 主程序入口
if __name__ == '__main__':
    pass