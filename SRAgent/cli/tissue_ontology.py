# import
# 导入部分

# 标准库导入
import os
import asyncio

# 第三方库导入
from Bio import Entrez
from langchain_core.messages import HumanMessage

# 项目内部模块导入
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.tissue_ontology import create_tissue_ontology_workflow
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_agent_stream, display_final_results

# 函数定义
def tissue_ontology_parser(subparsers):
    """
    配置组织本体代理的命令行参数解析器。

    Args:
        subparsers: argparse 的子解析器对象，用于添加新的子命令。
    """
    help_msg = '组织本体: 使用 Uberon 本体对组织描述进行分类。'
    description = """
    # 示例提示:
    1. "对以下组织进行分类: brain"
    2. "海马体的 Uberon ID 是什么?"
    3. "组织: 肺, 心脏, 肝脏"
    4. "查找肺部肺泡内衬上皮细胞薄层的本体术语"
    5. "骨骼肌组织的 Uberon 分类是什么?"
    """
    sub_parser = subparsers.add_parser(
        'tissue-ontology', help=help_msg, description=description, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=tissue_ontology_main)
    sub_parser.add_argument('prompt', type=str, help='要分类的组织描述')
    sub_parser.add_argument('--max-concurrency', type=int, default=3, 
                            help='最大并发进程数')
    sub_parser.add_argument('--recursion-limit', type=int, default=40,
                            help='最大递归限制')
    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='将工作流图写入文件并退出（支持 .png, .svg, .pdf, .mermaid 格式）'
    )
    
def tissue_ontology_main(args):
    """
    调用组织本体工作流的主函数。

    Args:
        args: 命令行解析后的参数对象。
    """
    # 设置 Entrez 邮箱和 API 密钥
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
    # 处理写入图选项
    if args.write_graph:
        handle_write_graph_option(create_tissue_ontology_workflow, args.write_graph)
        return

    # 以流式方式调用工作流
    config = {
        "max_concurrency": args.max_concurrency,
        "recursion_limit": args.recursion_limit
    }
    input_data = {"messages": [HumanMessage(content=args.prompt)]}
    results = asyncio.run(
        create_agent_stream(
            input_data, create_tissue_ontology_workflow, config, 
            summarize_steps=not args.no_summaries,
            no_progress=args.no_progress
        )
    )
    
    # 以富文本格式显示最终结果
    display_final_results(results, "🧬 Uberon Tissue Classifications 🧬")

# 主程序入口
if __name__ == '__main__':
    pass
