import unittest
import os
import tempfile
import shutil
import sys
from unittest.mock import patch, mock_open, MagicMock
from main import (
    calculate_similarity,
    read_file,
    write_result,
    main
)
from sklearn.feature_extraction.text import TfidfVectorizer


class TestPlagiarismChecker(unittest.TestCase):
    """论文查重系统单元测试"""

    def setUp(self):
        """每个测试前的准备工作"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """每个测试后的清理工作"""
        # 删除临时目录
        shutil.rmtree(self.test_dir)


    # 测试 calculate_similarity 函数
    def test_calculate_similarity_identical_texts(self):
        """测试完全相同文本的相似度计算"""
        text1 = "它们在胸前划着十字，一边谴责同类的这种行为，一边乞求上帝饶恕他们。"
        text2 = "它们在胸前划着十字，一边谴责同类的这种行为，一边乞求上帝饶恕他们。"
        similarity = calculate_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)

    def test_calculate_similarity_completely_different(self):
        """测试完全不同文本的相似度计算"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "明天是星期一，天气雨，我明天要去上学。"
        similarity = calculate_similarity(text1, text2)
        self.assertLess(similarity, 0.3)
    # 测试 calculate_similarity 函数
    def test_calculate_similarity_identical_texts(self):
        """测试完全相同文本的相似度计算"""
        text1 = "它们在胸前划着十字，一边谴责同类的这种行为，一边乞求上帝饶恕他们。"
        text2 = "它们在胸前划着十字，一边谴责同类的这种行为，一边乞求上帝饶恕他们。"
        similarity = calculate_similarity(text1, text2)
        self.assertEqual(similarity, 1.0)


    def test_calculate_similarity_empty_texts(self):
        """测试空文本的相似度计算"""
        # 1. 两个空文本
        similarity = calculate_similarity("", "")
        self.assertEqual(similarity, 1.0)

        # 2. 一个空文本，一个非空文本
        test_text_path = os.path.join(self.test_dir, "test_text.txt")
        with open(test_text_path, 'w', encoding='utf-8') as f:
            f.write("测试非空文本内容")
        empty_text_path = os.path.join(self.test_dir, "empty_text.txt")
        with open(empty_text_path, 'w', encoding='utf-8') as f:
            pass  # 空文件

        test_text = read_file(test_text_path)
        empty_text = read_file(empty_text_path)
        similarity = calculate_similarity(test_text, empty_text)
        self.assertEqual(similarity, 0.0)

    def test_calculate_similarity_whitespace_only(self):
        """测试只有空格的文本的相似度计算"""
        similarity = calculate_similarity("   ", "    ")
        self.assertEqual(similarity, 1.0)

    def test_calculate_similarity_special_characters(self):
        """测试包含特殊字符的文本的相似度计算"""
        text1 = "Python是一种广泛使用的高级编程语言，具有简洁、易读的语法。"
        text2 = "Python是一种广泛使用的高级编程语言，具有简洁、易读的语法！"
        similarity = calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.9)

    def test_calculate_similarity_long_texts(self):
        """测试长文本的相似度计算"""
        text1 = "自然语言处理是人工智能领域中的一个重要方向。" * 10
        text2 = "自然语言处理是AI领域中的一个重要方向。" * 10
        similarity = calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.6)

    # 测试 read_file 函数
    def test_read_file_exists(self):
        """测试读取存在的文件"""
        test_content = "测试文件内容"
        test_file = os.path.join(self.test_dir, "test.txt")

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        content = read_file(test_file)
        self.assertEqual(content, test_content)

    def test_read_file_not_exists(self):
        """测试读取不存在的文件"""
        test_file = os.path.join(self.test_dir, "nonexistent.txt")

        with self.assertRaises(Exception) as context:
            read_file(test_file)

        self.assertTrue("文件不存在" in str(context.exception))

    def test_read_file_no_permission(self):
        """测试读取没有权限的文件"""
        if os.name == 'posix':  # Unix/Linux/MacOS
            test_file = os.path.join(self.test_dir, "no_permission.txt")

            with open(test_file, 'w') as f:
                f.write("测试内容")

            # 移除读取权限
            os.chmod(test_file, 0o000)

            with self.assertRaises(Exception) as context:
                read_file(test_file)

            self.assertTrue("无法读取文件" in str(context.exception))

            # 恢复权限以便清理
            os.chmod(test_file, 0o644)

    def test_read_file_different_encodings(self):
        """测试读取不同编码的文件"""
        test_content = "测试文件内容"

        # 测试UTF-8编码
        test_file_utf8 = os.path.join(self.test_dir, "test_utf8.txt")
        with open(test_file_utf8, 'w', encoding='utf-8') as f:
            f.write(test_content)

        content = read_file(test_file_utf8)
        self.assertEqual(content, test_content)

        # 测试GBK编码
        test_file_gbk = os.path.join(self.test_dir, "test_gbk.txt")
        with open(test_file_gbk, 'w', encoding='gbk') as f:
            f.write(test_content)

        content = read_file(test_file_gbk)
        self.assertEqual(content, test_content)

    def test_read_file_empty_file(self):
        """测试读取空文件"""
        test_file = os.path.join(self.test_dir, "empty.txt")

        with open(test_file, 'w') as f:
            pass  # 创建空文件

        content = read_file(test_file)
        self.assertEqual(content, "")

    def test_read_file_large_file(self):
        """测试读取大文件"""
        test_content = "测试内容" * 10000  # 创建大约80KB的内容
        test_file = os.path.join(self.test_dir, "large.txt")

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        content = read_file(test_file)
        self.assertEqual(content, test_content)

    # 测试 write_result 函数
    def test_write_result_normal(self):
        """测试正常写入结果文件"""
        test_file = os.path.join(self.test_dir, "result.txt")
        similarity = 0.85

        write_result(test_file, similarity)

        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertEqual(content, "0.85")

    def test_write_result_create_directory(self):
        """测试创建不存在的目录并写入文件"""
        test_dir = os.path.join(self.test_dir, "subdir")
        test_file = os.path.join(test_dir, "result.txt")
        similarity = 0.75

        # 确保目录不存在
        self.assertFalse(os.path.exists(test_dir))

        write_result(test_file, similarity)

        # 检查目录和文件是否创建
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.exists(test_file))

        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertEqual(content, "0.75")

    def test_write_result_no_permission(self):
        """测试写入没有权限的目录"""
        if os.name == 'posix':  # Unix/Linux/MacOS
            test_dir = os.path.join(self.test_dir, "no_write_permission")
            os.makedirs(test_dir)

            # 移除写入权限
            os.chmod(test_dir, 0o444)

            test_file = os.path.join(test_dir, "result.txt")

            with self.assertRaises(Exception) as context:
                write_result(test_file, 0.85)

            self.assertTrue("无法写入文件" in str(context.exception))

            # 恢复权限以便清理
            os.chmod(test_dir, 0o755)

    def test_write_result_invalid_similarity(self):
        """测试写入无效的相似度值"""
        test_file = os.path.join(self.test_dir, "result.txt")

        # 测试小于0的值
        with self.assertRaises(Exception) as context:
            write_result(test_file, -0.1)
        self.assertTrue("无效的相似度值" in str(context.exception))

        # 测试大于1的值
        with self.assertRaises(Exception) as context:
            write_result(test_file, 1.1)
        self.assertTrue("无效的相似度值" in str(context.exception))

        # 测试非数字值
        with self.assertRaises(Exception) as context:
            write_result(test_file, "invalid")
        self.assertTrue("无效的相似度值" in str(context.exception))

    # 测试 main 函数和命令行参数处理
    @patch('main.read_file')
    @patch('main.write_result')
    @patch('main.calculate_similarity')
    def test_main_normal_execution(self, mock_calculate, mock_write, mock_read):
        """测试主函数正常执行"""
        # 设置模拟返回值
        mock_read.side_effect = ["原文内容", "抄袭版内容"]
        mock_calculate.return_value = 0.82

        # 模拟命令行参数
        test_args = ["main.py", r"D:\软工\测试文本(1)\orig.txt", r"D:\软工\测试文本(1)\orig_0.8_add.txt", "result.txt"]

        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                main()

        # 验证函数调用
        self.assertEqual(mock_read.call_count, 2)
        mock_calculate.assert_called_once_with("原文内容", "抄袭版内容")
        mock_write.assert_called_once_with("result.txt", 0.82)
        mock_exit.assert_not_called()  # 正常执行不应调用exit

    @patch('main.read_file')
    def test_main_file_not_exists(self, mock_read):
        """测试主函数处理文件不存在的情况"""
        # 模拟文件不存在异常
        mock_read.side_effect = Exception("文件不存在: orig.txt")

        # 模拟命令行参数
        test_args = ["main.py", "orig.txt", "plagiarized.txt", "result.txt"]

        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    main()

        # 验证错误处理和退出
        mock_exit.assert_called_once_with(1)
        mock_print.assert_called_once()
        self.assertTrue("错误" in mock_print.call_args[0][0])

    def test_main_insufficient_arguments(self):
        """测试主函数处理参数不足的情况"""
        # 模拟参数不足
        test_args = ["main.py", "orig.txt", "plagiarized.txt"]

        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    main()

        # 验证错误处理和退出
        mock_exit.assert_called_once_with(1)
        mock_print.assert_called_once()
        self.assertTrue("错误" in mock_print.call_args[0][0])

    @patch('main.read_file')
    def test_main_large_file(self, mock_read):
        """测试主函数处理大文件的情况"""
        # 模拟大文件异常
        mock_read.side_effect = Exception("原文文件过大: 10.50MB > 10MB")

        # 模拟命令行参数
        test_args = ["main.py", "orig.txt", "plagiarized.txt", "result.txt"]

        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    main()

        # 验证错误处理和退出
        mock_exit.assert_called_once_with(1)
        mock_print.assert_called_once()
        self.assertTrue("错误" in mock_print.call_args[0][0])

    # 测试错误处理路径
    @patch('main.read_file')
    @patch('main.calculate_similarity')
    def test_main_calculation_error(self, mock_calculate, mock_read):
        """测试主函数处理计算错误的情况"""
        # 设置模拟返回值
        mock_read.side_effect = ["原文内容", "抄袭版内容"]
        mock_calculate.side_effect = Exception("相似度计算失败")

        # 模拟命令行参数
        test_args = ["main.py", "orig.txt", "plagiarized.txt", "result.txt"]

        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    main()

        # 验证错误处理和退出
        mock_exit.assert_called_once_with(1)
        mock_print.assert_called_once()
        self.assertTrue("错误" in mock_print.call_args[0][0])

    @patch('main.read_file')
    @patch('main.calculate_similarity')
    @patch('main.write_result')
    def test_main_write_error(self, mock_write, mock_calculate, mock_read):
        """测试主函数处理写入错误的情况"""
        # 设置模拟返回值
        mock_read.side_effect = ["原文内容", "抄袭版内容"]
        mock_calculate.return_value = 0.82
        mock_write.side_effect = Exception("无法写入文件: result.txt")

        # 模拟命令行参数
        test_args = ["main.py", "orig.txt", "plagiarized.txt", "result.txt"]

        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    main()

        # 验证错误处理和退出
        mock_exit.assert_called_once_with(1)
        mock_print.assert_called_once()
        self.assertTrue("错误" in mock_print.call_args[0][0])


    # # --------------------------
    # # 补充测试极端文本场景
    # # --------------------------
    # def test_calculate_similarity_large_text(self):
    #     """测试大文本（接近10MB）的相似度计算"""
    #     # 生成接近10MB的重复文本
    #     large_text = "重复内容 " * (1024 * 1024)  # 约8MB（每个"重复内容 "约8字节）
    #     similar_text = large_text + " 新增少量内容"
    #
    #     similarity = calculate_similarity(large_text, similar_text)
    #     self.assertGreater(similarity, 0.9)  # 高度相似



if __name__ == '__main__':
    unittest.main()