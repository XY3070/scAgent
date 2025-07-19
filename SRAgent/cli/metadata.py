# import
# 导入标准库
import sys
import asyncio
import argparse
from typing import List

# 导入第三方库
import pandas as pd
from Bio import Entrez
from langchain_core.messages import HumanMessage

# 导入项目内部模块
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.metadata import get_metadata_items, create_metadata_graph
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_step_summary_chain
from SRAgent.db.get import db_get_unprocessed_records


# 定义函数
def metadata_agent_parser(subparsers):
    """
    为元数据代理设置命令行参数解析器。

    Args:
        subparsers: argparse 的子解析器对象，用于添加新的子命令。

    Returns:
        None
    """
    help_msg = '元数据代理：获取特定 SRX 登录号的元数据'
    description = """
    此命令用于从各种来源（如 CSV 文件或数据库）获取 SRX 登录号的元数据。
    它支持并发处理、结果过滤和输出到不同格式的文件。
    """
    sub_parser = subparsers.add_parser(
        'metadata', help=help_msg, description=description, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=metadata_agent_main)
    sub_parser.add_argument(
        'srx_accession_csv', type=str, nargs='?', default=None,
        help='包含 entrez_id 和 srx_accession 的 CSV 文件路径。需要包含标题行。当使用 --from-db 选项时，此参数可选。'
    )    
    sub_parser.add_argument(
        '--from-db', action='store_true', default=False,
        help='从数据库而不是 CSV 文件获取 SRX 登录号。'
    )
    sub_parser.add_argument(
        '--database', type=str, default='sra', choices=['gds', 'sra'], 
        help='Entrez ID 的 Entrez 数据库来源。可选值有 "gds" 和 "sra"。'
    )
    sub_parser.add_argument(
        '--max-concurrency', type=int, default=6, help='最大并发进程数。'
    )
    sub_parser.add_argument(
        '--recursion-limit', type=int, default=200, help='最大递归限制。'
    )
    sub_parser.add_argument(
        '--max-parallel', type=int, default=2, help='SRX 登录号的最大并行处理数。'
    )
    sub_parser.add_argument(
        '--no-srr', action='store_true', default=False, 
        help='不将 SRR 登录号上传到 scBaseCount SQL 数据库。'
    )
    sub_parser.add_argument(
        '--use-database', action='store_true', default=False, 
        help='将结果添加到 scBaseCount SQL 数据库。'
    )

    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='将工作流图写入文件并退出（支持 .png, .svg, .pdf, .mermaid 格式）。'
    )
    sub_parser.add_argument(
        '--output-csv', type=str, default=None,
        help='保存元数据为 CSV 文件的路径。'
    )
    sub_parser.add_argument(
        '--output-file', type=str, default=None,
        help='保存元数据为 JSON 文件的路径。'
    )
    sub_parser.add_argument(
        '--limit', type=int, default=None,
        help='限制处理的 SRX 登录号数量。'
    )
    sub_parser.add_argument(
        '--filter-by', type=str, action='append', default=[],
        help='通过键值对过滤结果（例如：organism=Homo sapiens）。可以指定多次。'
    )
    sub_parser.add_argument(
        '--no-summaries', action='store_true', default=False,
        help='处理过程中不打印步骤摘要。'
    )
    sub_parser.add_argument(
        '--query', type=str, default=None,
        help='用于过滤结果的查询字符串（例如：“human lung”）。此参数将被转换为 --filter-by 参数。'
    )
    # 添加 organism, single_cell 和 keywords 参数
    sub_parser.add_argument(
        '--organism', type=str, default=None,
        help='按生物体过滤 SRX 登录号。'
    )
    sub_parser.add_argument(
        '--single-cell', type=str, default=None, choices=['true', 'false'],
        help='按单细胞状态过滤 SRX 登录号（“true”或“false”）。'
    )
    sub_parser.add_argument(
        '--keywords', type=str, default=None,
        help='按关键词过滤 SRX 登录号。'
    )


async def _process_single_srx(
    entrez_srx, database, graph, step_summary_chain, config: dict, no_summaries: bool
):
    """
    处理单个 Entrez ID 对应的 SRX 访问号。

    Args:
        entrez_srx: 包含 Entrez ID 和 SRX 访问号的元组。
        database: Entrez 数据库名称。
        graph: 用于处理元数据的工作流图。
        step_summary_chain: 用于生成步骤摘要的链。
        config: 包含配置参数的字典。
        no_summaries: 布尔值，指示是否打印步骤摘要。

    Returns:
        提取的元数据字典。
    """
    # 格式化图的输入
    metadata_items = "\n".join([f" - {x}" for x in get_metadata_items().values()])
    prompt = "\n".join([
        "# 指令",
        "对于 SRA 实验登录号 {SRX_accession}，查找以下数据集元数据:",
        metadata_items,
        "# 注意",
        " - 尝试用两个数据源确认任何可疑的元数据值"
    ])
    input_data = {
        "entrez_id": entrez_srx[0],
        "SRX": entrez_srx[1],
        "database": database,
        "messages": [HumanMessage(prompt.format(SRX_accession=entrez_srx[1]))]
    }

    # 调用图
    final_state = None
    i = 0
    async for step in graph.astream(input_data, config=config):
        i += 1
        final_state = step
        if no_summaries:
            nodes = ",".join(list(step.keys()))
            print(f"[{entrez_srx[0]}] 步骤 {i}: {nodes}")
        else:
            msg = await step_summary_chain.ainvoke({"step": step})
            print(f"[{entrez_srx[0]}] 步骤 {i}: {msg.content}")

    # 初始化提取的元数据字典
    extracted_metadata = {
        "entrez_id": entrez_srx[0],
        "SRX": entrez_srx[1],
        "is_illumina": None,
        "is_single_cell": None,
        "is_paired_end": None,
        "lib_prep": None,
        "tech_10x": None,
        "cell_prep": None,
        "organism": None,
        "tissue": None,
        "disease": None,
        "perturbation": None,
        "cell_line": None,
        "tissue_ontology_term_id": None,
        "SRR": None,
    }

    # 从最终状态中提取元数据
    if final_state:
        try:
            extracted_metadata.update({
                "is_illumina": final_state.get("is_illumina"),
                "is_single_cell": final_state.get("is_single_cell"),
                "is_paired_end": final_state.get("is_paired_end"),
                "lib_prep": final_state.get("lib_prep"),
                "tech_10x": final_state.get("tech_10x"),
                "cell_prep": final_state.get("cell_prep"),
                "organism": final_state.get("organism"),
                "tissue": final_state.get("tissue"),
                "disease": final_state.get("disease"),
                "perturbation": final_state.get("perturbation"),
                "cell_line": final_state.get("cell_line"),
                "tissue_ontology_term_id": ", ".join(final_state.get("tissue_ontology_term_id", []) if final_state.get("tissue_ontology_term_id") is not None else []),
                "SRR": ", ".join(final_state.get("SRR", []) if final_state.get("SRR") is not None else []),
            })
            # 提取后打印最终结果，但确保返回 extracted_metadata
            try:
                print(f"#-- Entrez ID {entrez_srx[0]} 的最终结果 --#")
                print(final_state["final_state_node"]["messages"][-1].content)
                print("#---------------------------------------------#")
            except KeyError:
                print(f"#-- Entrez ID {entrez_srx[0]} 的最终结果 --#")
                print("无法检索最终消息内容。")
                print("#---------------------------------------------#")
        except Exception as e:
            print(f"从最终状态提取元数据时出错: {e}")
            print("#---------------------------------------------#")
    return extracted_metadata



async def _metadata_agent_main(args):
    """
    调用元数据代理的主函数。

    Args:
        args: 命令行解析后的参数对象。

    Returns:
        None
    """
    try:
        print(f"解析的参数: {args}")

        # 如果提供了 --query，则将其转换为 --filter-by
        if args.query:
            args.filter_by.append(f"query={args.query}")
        
        # 处理写入图选项
        if args.write_graph:
            handle_write_graph_option(create_metadata_graph, args.write_graph)
            return

        # 创建监督代理
        graph = create_metadata_graph()
        step_summary_chain = create_step_summary_chain()

        # 调用代理
        config = {
            "max_concurrency": args.max_concurrency,
            "recursion_limit": args.recursion_limit,
            "configurable": {
                "use_database": args.use_database,
                "no_srr": args.no_srr
            }
        }

        # 根据参数从数据库或 CSV 文件读取 entrez_id 和 srx_accession
        if args.from_db:
            from SRAgent.db.connect import db_connect
            from SRAgent.db.get import db_get_unprocessed_records, db_get_filtered_srx_metadata
            with db_connect() as conn:
                # 如果提供了过滤器，则使用 db_get_filtered_srx_metadata
                records_df = db_get_filtered_srx_metadata(
                    conn=conn,
                    organism=args.organism,
                    is_single_cell=args.single_cell,
                    keywords=args.keywords,
                    query=args.query,
                    limit=args.limit,
                    database=args.database
                )
                entrez_srx_accessions = list(records_df[['entrez_id', 'srx_accession']].itertuples(index=False, name=None))
                
                print(f"从数据库中检索到的记录: {entrez_srx_accessions}")
                if not entrez_srx_accessions:
                    print("数据库中没有找到未处理的记录。")
                    return
                print("已到达 db_connect 块的末尾。")
        else:
            if not args.srx_accession_csv:
                raise ValueError("必须提供 --from-db 或 srx_accession_csv。")
            # 从 CSV 文件中读取 entrez_id 和 srx_accession
            df_csv = pd.read_csv(args.srx_accession_csv)
            entrez_srx_accessions = list(df_csv[['entrez_id', 'srx_accession']].itertuples(index=False, name=None))

        # 限制处理的 SRX 登录号数量
        if args.limit:
            entrez_srx_accessions = entrez_srx_accessions[:args.limit]

        # 创建信号量以限制并发处理
        semaphore = asyncio.Semaphore(args.max_parallel)

        async def _process_with_semaphore(entrez_srx):
            async with semaphore:
                return await _process_single_srx(
                    entrez_srx,
                    args.database,
                    graph,
                    step_summary_chain,
                    config,
                    args.no_summaries
                )

        # 并发处理所有 SRX 登录号
        raw_results = await asyncio.gather(*[_process_with_semaphore(esrx) for esrx in entrez_srx_accessions])
        valid_results = [res for res in raw_results if res is not None]

        # 如果没有收集到元数据，则打印消息并返回
        if not valid_results:
            print("没有收集到元数据。")
            return

        # 收集所有 SRR 访问号
        all_srr_accessions = []
        for item in valid_results:
            if item.get("SRR"):
                srx_acc = item.get("SRX")
                srr_list = [s.strip() for s in item["SRR"].split(", ") if s.strip()]
                for srr in srr_list:
                    all_srr_accessions.append({"srx_accession": srx_acc, "srr_accession": srr})

        # 将结果转换为 DataFrame
        df = pd.DataFrame(valid_results)
        print(f"DataFrame 形状: {df.shape}")
        print(f"DataFrame 头部:\n{df.head().to_string()}")

        # 如果提供了过滤器（仅适用于 CSV 输入，数据库过滤器已由 db_get_filtered_srx_metadata 处理），则应用过滤器
        if not args.from_db and args.filter_by:
            for filter_str in args.filter_by:
                try:
                    key, value = filter_str.split('=', 1)
                    if key in df.columns:
                        # 将列转换为字符串进行比较
                        df = df[df[key].astype(str) == value]
                    else:
                        print(f"警告: 元数据中未找到过滤键 '{key}'。跳过过滤器。")
                except ValueError:
                    print(f"警告: 过滤器格式 '{filter_str}' 无效。预期格式为 'key=value'。跳过过滤器。")



        # 输出结果
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"元数据已成功保存到 {args.output_csv}")
        elif args.output_file:
            df.to_json(args.output_file, orient='records', indent=2)
            print(f"元数据已成功保存到 {args.output_file}")

        # 添加到数据库
        if args.use_database:
            from SRAgent.db.connect import db_connect
            from SRAgent.db.add import db_add_srx_metadata, db_add_srr_accessions
            with db_connect() as session:
                for item in valid_results:
                    db_add_srx_metadata(session, item)
                    if not args.no_srr and item.get('SRR'):
                        srr_accessions = [s.strip() for s in item['SRR'].split(',') if s.strip()]
                        db_add_srr_accessions(session, item['SRX'], srr_accessions)
                session.commit()
            print("元数据已添加到数据库。")

    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

def metadata_agent_main(args):
    """
    元数据代理的同步入口点。
    此函数是命令行接口的入口，它调用异步主函数。
    """
    asyncio.run(_metadata_agent_main(args))


# main
if __name__ == '__main__':
    pass