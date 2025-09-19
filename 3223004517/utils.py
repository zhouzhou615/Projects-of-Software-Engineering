def read_file(file_path):
    """读取文件函数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def write_result(file_path, similarity):
    """写入文件函数"""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"{similarity:.2f}")
