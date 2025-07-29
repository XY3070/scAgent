#!/usr/bin/env python
# 导入必要的库

# 标准库
import os
import sys
import argparse

# 第三方库
from dotenv import load_dotenv

# 项目内部模块
from SRAgent.cli.utils import CustomFormatter
from SRAgent.cli.entrez import entrez_agent_parser, entrez_agent_main
from SRAgent.cli.sragent import sragent_parser, sragent_main
from SRAgent.cli.metadata import metadata_agent_parser, metadata_agent_main
from SRAgent.cli.srx_info import SRX_info_agent_parser, SRX_info_agent_main
from SRAgent.cli.find_datasets import find_datasets_parser, find_datasets_main
from SRAgent.cli.tissue_ontology import tissue_ontology_parser, tissue_ontology_main
from SRAgent.db.connect import db_connect
from SRAgent.db.create import create_srx_metadata, create_srx_srr
import psycopg2


# 函数定义
def arg_parse(args=None) -> dict:
    """
    解析命令行参数。
    """
    # 程序的描述信息
    desc = "SRAgent: 一个用于处理SRA的多智能体工具"
    # 程序的详细描述信息
    epi = """DESCRIPTION:
    SRAgent是一个多智能体工具，用于处理Sequence Read Archive (SRA)数据库和其他Entrez数据库。
    它旨在成为一个灵活且易于使用的工具，用于与SRA和Entrez进行交互。
    """
    
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description=desc,  # 程序描述
        epilog=epi,        # 程序的详细描述
        formatter_class=CustomFormatter  # 使用自定义的格式化类
    )

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 添加Entrez智能体子命令
    entrez_agent_parser(subparsers)
    # 添加SR智能体子命令
    sragent_parser(subparsers)
    # 添加Metadata智能体子命令
    metadata_agent_parser(subparsers)
    # 添加Tissue Ontology子命令
    tissue_ontology_parser(subparsers)
    # 添加SRX信息智能体子命令
    SRX_info_agent_parser(subparsers)
    # 添加查找数据集子命令
    find_datasets_parser(subparsers)
    
    # 解析并返回命令行参数
    return parser.parse_args()

def main():
    # 加载环境变量，如果存在同名变量则覆盖
    load_dotenv(override=True)
    # 解析命令行参数
    args = arg_parse()

    # 尝试连接数据库并创建表（如果不存在）
    try:
        with db_connect() as conn:
            # 创建srx_metadata表
            create_srx_metadata(conn)
            # 创建srx_srr表
            create_srx_srr(conn)
            # 提交数据库更改
            conn.commit()

    except Exception as e:
         # 如果连接数据库或创建表失败，则打印错误信息并退出
         print(f"连接数据库或创建表时发生错误: {e}")
         sys.exit(1)

    # 根据子命令执行相应的操作
    if not args.command:
        # 如果没有指定子命令，则打印提示信息并退出
        print("请提供一个子命令，或使用 -h/--help 获取帮助")
        sys.exit(0)
    elif args.command.lower() == "entrez":
        # 执行Entrez智能体主函数
        entrez_agent_main(args)
    elif args.command.lower() == "sragent":
        # 执行SRAgent智能体主函数
        sragent_main(args)
    elif args.command.lower() == "metadata":
        # 执行Metadata智能体主函数
        metadata_agent_main(args)
    elif args.command.lower() == "tissue-ontology":
        # 执行Tissue Ontology主函数
        tissue_ontology_main(args)
    elif args.command.lower() == "srx-info":
        # 执行SRX信息智能体主函数
        SRX_info_agent_main(args)
    elif args.command.lower() == "find-datasets":
        # 执行查找数据集主函数
        if args.tenant:
            os.environ["DYNACONF_ENV"] = args.tenant
        with db_connect() as conn_for_find_datasets:
            find_datasets_main(args, conn_for_find_datasets)
    else:
        # 如果指定了未知命令，则打印提示信息并退出
        print("未指定命令。正在退出...")
        sys.exit(0)

# 如果作为主程序运行，则调用main函数
if __name__ == "__main__":
    main()