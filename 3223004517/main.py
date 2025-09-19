import argparse
import os
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

    try:
        # 使用jieba进行中文分词
        original_words = ' '.join(jieba.cut(original_text))
        plagiarized_words = ' '.join(jieba.cut(plagiarized_text))
    except Exception as e:
        raise Exception(f"分词处理失败: {str(e)}")

    try:
        # 使用TF-IDF向量化文本
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([original_words, plagiarized_words])
    except Exception as e:
        raise Exception(f"文本向量化失败: {str(e)}")

    try:
        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # 处理可能的NaN值
        if np.isnan(similarity):
            return 0.0

        return round(similarity, 2)
    except Exception as e:
        raise Exception(f"相似度计算失败: {str(e)}")

def main():
    print("论文查重系统 - 批量处理（结果汇总到同一文件）")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='论文查重系统')
    parser.add_argument(
        '--original',
        type=str,
        default=r"D:\软工\测试文本(1)\orig.txt",  # 原始论文默认路径
        help='原始论文文件路径'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=r'D:\软工\paper_check\3223004517\result.txt',  # 统一结果文件
        help='所有查重结果的统一输出文件路径'
    )
    args = parser.parse_args()

    # 确保输出文件的目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义需要检测的“抄袭文件列表”
    plagiarized_files = [
        r"D:\软工\测试文本(1)\orig_0.8_add.txt",
        r"D:\软工\测试文本(1)\orig_0.8_del.txt",
        r"D:\软工\测试文本(1)\orig_0.8_dis_1.txt",
        r"D:\软工\测试文本(1)\orig_0.8_dis_10.txt",
        r"D:\软工\测试文本(1)\orig_0.8_dis_15.txt"
    ]

    # 读取原始文件内容
    try:
        original_text = read_file(args.original)
    except Exception as e:
        print(f"原始文件读取失败: {str(e)}")
        sys.exit(1)

    # 原始文件大小限制（防止内存溢出）
    max_file_size = 10 * 1024 * 1024  # 10MB
    if os.path.getsize(args.original) > max_file_size:
        print(f"原始文件过大（{os.path.getsize(args.original) / 1024 / 1024:.2f}MB > 10MB）")
        sys.exit(1)

    # 遍历每个抄袭文件，计算重复率并追加到统一结果文件
    for plag_file in plagiarized_files:
        try:
            # 检查抄袭文件大小
            if os.path.getsize(plag_file) > max_file_size:
                print(f"跳过 {plag_file}：文件过大（{os.path.getsize(plag_file) / 1024 / 1024:.2f}MB > 10MB）")
                # 追加“跳过记录”到结果文件
                with open(args.output_file, 'a', encoding='utf-8') as f:
                    f.write(f"文件: {plag_file}, 状态: 跳过（文件过大）\n")
                continue

            # 读取抄袭文件内容
            plagiarized_text = read_file(plag_file)
            if plagiarized_text is None:
                # 追加“读取失败记录”到结果文件
                with open(args.output_file, 'a', encoding='utf-8') as f:
                    f.write(f"文件: {plag_file}, 状态: 读取失败\n")
                continue

            # 计算相似度
            similarity = calculate_similarity(original_text, plagiarized_text)

            with open(args.output_file, 'a', encoding='utf-8') as f:
                f.write(f"文件: {plag_file}, 重复率: {similarity:.2f}\n")

            print(f"✅ 处理完成：{plag_file} → 重复率 {similarity:.2f}，结果已追加到 {args.output_file}")

        except Exception as e:
            # 追加“错误记录”到结果文件
            with open(args.output_file, 'a', encoding='utf-8') as f:
                f.write(f"文件: {plag_file}, 状态: 出错（{str(e)}）\n")
            print(f"❌ 处理 {plag_file} 出错: {str(e)}")



if __name__ == '__main__':
    main()

