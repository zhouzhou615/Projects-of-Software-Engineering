import os
def read_file(file_path):
    """读取文件函数"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise Exception(f"文件不存在: {file_path}")

    # 检查文件是否可读
    if not os.access(file_path, os.R_OK):
        raise Exception(f"无法读取文件: {file_path}")

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
            raise Exception(f"文件编码不支持: {file_path}")


def write_result(file_path, similarity):
    """写入文件函数"""
    # 检查目录是否存在
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 检查文件是否可写
    if os.path.exists(file_path) and not os.access(file_path, os.W_OK):
        raise Exception(f"无法写入文件: {file_path}")

    # 写入结果
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"{similarity:.2f}")