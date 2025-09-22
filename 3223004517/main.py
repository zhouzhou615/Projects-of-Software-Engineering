import argparse
import sys
import os
import functools
import re
from utils import read_file, write_result


# 添加文件内容缓存
@functools.lru_cache(maxsize=32)
def cached_read_file(file_path):
    """带缓存的文件读取函数"""
    return read_file(file_path)


# 添加分词结果缓存
@functools.lru_cache(maxsize=100)
def cached_cut_text(text):
    """带缓存的分词函数 - 针对Windows优化"""
    import jieba
    # 在Windows上不使用并行模式
    return ' '.join(jieba.cut(text))


def preprocess_text(text):
    """
    文本预处理函数
    清洗文本，移除标点符号和多余空格
    """
    # 移除标点符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 移除首尾空格
    return text.strip()


def calculate_similarity(original_text, plagiarized_text, fast_mode=False):
    """
    计算两段文本的相似度
    使用TF-IDF向量化和余弦相似度算法 - Windows优化版
    """
    # 处理空文本情况
    if not original_text.strip() and not plagiarized_text.strip():
        return 1.0

    if not original_text.strip() or not plagiarized_text.strip():
        return 0.0

    # 文本预处理
    original_text = preprocess_text(original_text)
    plagiarized_text = preprocess_text(plagiarized_text)

    # 检查预处理后的文本是否为空
    if not original_text or not plagiarized_text:
        return 0.0

    # 延迟导入第三方库
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        import jieba
        # 精确模式分词 + 去除分词后空字符串
        original_words = [word for word in jieba.cut(original_text, cut_all=False) if word.strip()]
        plagiarized_words = [word for word in jieba.cut(plagiarized_text, cut_all=False) if word.strip()]

        # 若分词后无有效词汇（极端情况）
        if not original_words or not plagiarized_words:
            return 0.0

        # 转换为空格分隔的字符串（适配TF-IDF输入格式）
        original_seg = ' '.join(original_words)
        plagiarized_seg = ' '.join(plagiarized_words)
    except Exception as e:
        raise Exception(f"分词处理失败: {str(e)}")

    try:
        # 配置TF-IDF参数：
        # - 过滤停用词（减少无意义词汇干扰）
        # - 包含1-gram和2-gram（捕捉短语级相似性）
        # - 过滤低频词（出现次数<2的词不纳入计算）
        stop_words = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '上', '也', '很',
                      '到', '说', '要', '去', '你']
        vectorizer = TfidfVectorizer(
            max_features=5000,  # 限制特征数量，提高性能
            min_df=1,  # 词至少在1个文档中出现
            max_df=1.0  # 词最多在100%的文档中出现
        )

        # 向量化处理
        tfidf_matrix = vectorizer.fit_transform([original_seg, plagiarized_seg])
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
    论文查重系统主函数 - Windows优化版
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
        default=r"D:\软工\测试文本(1)\orig_0.8_del.txt",  # 默认抄袭文件路径
        help='待检测论文文件路径'
    )
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=r'D:\软工\paper_check\3223004517\result.txt',  # 默认输出路径
        help='查重结果输出文件路径'
    )

    # 添加性能分析选项
    parser.add_argument(
        '--profile',
        action='store_true',
        help='启用性能分析'
    )


    try:
        args = parser.parse_args()

        # 验证文件路径
        if not all([args.original_file, args.plagiarized_file, args.output_file]):
            raise Exception("必须提供三个文件路径参数")

        # 检查文件是否存在
        for file_path in [args.original_file, args.plagiarized_file]:
            if not os.path.exists(file_path):
                raise Exception(f"文件不存在: {file_path}")

        # 检查文件大小限制（防止内存溢出）
        max_file_size = 10 * 1024 * 1024  # 10MB

        if os.path.getsize(args.original_file) > max_file_size:
            raise Exception(f"原文文件过大: {os.path.getsize(args.original_file) / 1024 / 1024:.2f}MB > 10MB")

        if os.path.getsize(args.plagiarized_file) > max_file_size:
            raise Exception(f"抄袭版文件过大: {os.path.getsize(args.plagiarized_file) / 1024 / 1024:.2f}MB > 10MB")

        # 读取文件内容 - 使用缓存版本
        original_text = cached_read_file(args.original_file)
        plagiarized_text = cached_read_file(args.plagiarized_file)

        # 性能分析开关
        if args.profile:
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            profiler.enable()

            # 计算相似度
            similarity = calculate_similarity(original_text, plagiarized_text)

            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumtime')
            stats.print_stats(20)  # 打印前20个最耗时的函数
        else:
            # 计算相似度
            similarity = calculate_similarity(original_text, plagiarized_text)

        # 写入结果
        write_result(args.output_file, similarity)

        print(f"查重完成，重复率为: {similarity:.4f}")

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()