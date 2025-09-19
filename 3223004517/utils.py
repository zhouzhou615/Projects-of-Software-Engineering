import os
def read_file(file_path):
    """读取文件函数"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise Exception(f"文件不存在: {file_path}")

        # 检查文件是否可读
        if not os.access(file_path, os.R_OK):
            raise Exception(f"无法读取文件: {file_path}")

        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return ""

        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as file:
                    content = file.read()
                return content
            except:
                # 尝试二进制读取
                try:
                    with open(file_path, 'rb') as file:
                        content = file.read()
                    # 尝试常见编码
                    for encoding in ['utf-8', 'gbk', 'gb2312', 'ascii']:
                        try:
                            return content.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                    raise Exception(f"文件编码不支持: {file_path}")
                except:
                    raise Exception(f"文件读取失败: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件失败: {str(e)}")



def write_result(file_path, similarity):
    """写入文件函数"""
    try:
        # 检查目录是否存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 检查文件是否可写
        if os.path.exists(file_path) and not os.access(file_path, os.W_OK):
            raise Exception(f"无法写入文件: {file_path}")

        # 验证相似度值
        if not isinstance(similarity, (int, float)) or similarity < 0 or similarity > 1:
            raise Exception(f"无效的相似度值: {similarity}")

        # 写入结果
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(f"{similarity:.2f}")
    except Exception as e:
        raise Exception(f"写入文件失败: {str(e)}")
