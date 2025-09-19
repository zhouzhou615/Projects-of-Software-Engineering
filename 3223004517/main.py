import argparse
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import read_file, write_result
import sys


def calculate_similarity(original_text, plagiarized_text):
    """计算两段文本的相似度"""
    # 如果两个文本都为空，则相似度为1.0
    if not original_text.strip() and not plagiarized_text.strip():
        return 1.0

    # 如果其中一个文本为空，则相似度为0.0
    if not original_text.strip() or not plagiarized_text.strip():
        return 0.0

    # 使用jieba进行中文分词
    original_words = ' '.join(jieba.cut(original_text))
    plagiarized_words = ' '.join(jieba.cut(plagiarized_text))

    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_words, plagiarized_words])

    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # 处理可能的NaN值
    if np.isnan(similarity):
        return 0.0

    return round(similarity, 2)


def main():
    print("论文查重系统 - 核心算法已实现")

    # 配置命令行参数（支持默认值，也可手动传入）
    parser = argparse.ArgumentParser(description='论文查重系统')
    parser.add_argument(
        'original_file',
        type=str,
        nargs='?',  # 允许参数可选
        default=r"D:\软工\测试文本(1)\orig.txt",  # 默认原文路径
        help='原始论文文件路径'
    )
    parser.add_argument(
        'plagiarized_file',
        type=str,
        nargs='?',
        default=r"D:\软工\测试文本(1)\orig_0.8_add.txt",  # 默认抄袭文件路径
        help='待检测论文文件路径'
    )
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=r'D:\软工\paper_check\3223004517\result.txt',  # 默认输出路径
        help='查重结果输出文件路径'
    )

    args = parser.parse_args()
    print(f"原文文件: {args.original_file}")
    print(f"抄袭版文件: {args.plagiarized_file}")
    print(f"输出文件: {args.output_file}")

    # 读取文件内容
    original_text = read_file(args.original_file)
    plagiarized_text = read_file(args.plagiarized_file)

    if original_text is None or plagiarized_text is None:
        print("程序终止：无法读取输入文件")
        return

    # 计算相似度并输出结果
    similarity = calculate_similarity(original_text, plagiarized_text)
    print(f"计算完成：两段文本的相似度为 {similarity}")
    write_result(args.output_file, similarity)
if __name__ == '__main__':
    main()

