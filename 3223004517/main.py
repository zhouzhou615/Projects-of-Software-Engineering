import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='论文查重系统')
    parser.add_argument('original_file', type=str, help='原文文件路径')
    parser.add_argument('plagiarized_file', type=str, help='抄袭版论文文件路径')
    parser.add_argument('output_file', type=str, help='输出答案文件路径')

    args = parser.parse_args()
    print(f"原文文件: {args.original_file}")
    print(f"抄袭版文件: {args.plagiarized_file}")
    print(f"输出文件: {args.output_file}")


if __name__ == '__main__':
    main()