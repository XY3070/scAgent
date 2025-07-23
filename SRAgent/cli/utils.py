import os
# 导入部分
import sys
import argparse

# 类定义
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    """
    自定义的帮助信息格式化器，结合了 ArgumentDefaultsHelpFormatter 和 RawDescriptionHelpFormatter 的功能。

    - ArgumentDefaultsHelpFormatter: 在帮助消息中显示参数的默认值。
    - RawDescriptionHelpFormatter: 允许在帮助消息中使用原始的、未格式化的描述文本，
                                 不自动换行和缩进，
                                 这对于多行描述或包含特殊格式的文本非常有用。
    """
    pass
