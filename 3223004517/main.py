import argparse
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from utils import read_file, write_result


def calculate_similarity(original_text, plagiarized_text):
    """
    计算两段文本的相似度
    使用TF-IDF向量化和余弦相似度算法
    """
    # 处理空文本情况
    if not original_text.strip() and not plagiarized_text.strip():
        return 1.0

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
    """
    论文查重系统主函数
    从命令行接收三个参数：原文文件、抄袭版论文文件、输出答案文件
    """
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
    try:
        args = parser.parse_args()

        # 验证文件路径
        if not all([args.original_file, args.plagiarized_file, args.output_file]):
            raise Exception("必须提供三个文件路径参数")

        # 检查文件大小限制（防止内存溢出）
        import os
        max_file_size = 10 * 1024 * 1024  # 10MB

        if os.path.getsize(args.original_file) > max_file_size:
            raise Exception(f"原文文件过大: {os.path.getsize(args.original_file) / 1024 / 1024:.2f}MB > 10MB")

        if os.path.getsize(args.plagiarized_file) > max_file_size:
            raise Exception(f"抄袭版文件过大: {os.path.getsize(args.plagiarized_file) / 1024 / 1024:.2f}MB > 10MB")

        # 读取文件内容
        original_text = read_file(args.original_file)
        plagiarized_text = read_file(args.plagiarized_file)

        # 计算相似度
        similarity = calculate_similarity(original_text, plagiarized_text)

        # 写入结果
        write_result(args.output_file, similarity)

        print(f"查重完成，重复率为: {similarity:.2f}")

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()