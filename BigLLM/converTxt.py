import os
import chardet


def detect_file_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到的编码: {encoding}, 置信度: {confidence}")
        return encoding


def convert_to_utf8(input_file, output_file, error_log):
    """将文件编码转换为 UTF-8，并处理错误"""
    try:
        # 检测文件编码
        encoding = detect_file_encoding(input_file)
        if encoding is None:
            raise ValueError(f"无法检测到编码，请手动检查文件: {input_file}")

        # 使用检测到的编码读取文件，并忽略/替换无法解码的字符
        with open(input_file, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()

        # 保存为 UTF-8 编码的新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"文件已成功转换为 UTF-8 编码，并保存到 {output_file}")

    except Exception as e:
        # 处理所有其他异常
        print(f"文件 {input_file} 转换时出现错误: {e}")
        with open(error_log, 'a', encoding='utf-8') as log:
            log.write(f"文件 {input_file} 转换失败: {str(e)}\n")


def process_directory(input_dir, output_dir, error_log):
    """扫描目录下所有文件，转换为 UTF-8 编码并保存到新目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_counter = 1
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            input_file_path = os.path.join(root, file_name)
            output_file_path = os.path.join(output_dir, f"{file_counter}.txt")

            # 执行文件转换
            convert_to_utf8(input_file_path, output_file_path, error_log)
            file_counter += 1


# 示例：处理目录中的所有文件
input_dir = './OrgFilm'  # 输入目录路径
output_dir = './transData'  # 输出目录路径
error_log = 'error_log.txt'  # 错误日志文件

# 调用函数处理目录
process_directory(input_dir, output_dir, error_log)