import json
import os
from sklearn.model_selection import train_test_split

# 保存数据为 .txt 文件，每行是一个压缩的 JSON 字符串
def save_as_txt(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            # 确保输出不使用 ASCII 编码，确保中文字符正常保存
            compressed_json = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
            f.write(compressed_json + '\n')

# 合并所有 JSON 文件为一个数据集
def merge_all_json_files(input_folder):
    all_data = []
    # 遍历输入文件夹中的所有 JSON 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_folder, file_name)
            # 打开并加载 JSON 数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)  # 合并所有数据
    return all_data

# 从合并的数据中随机划分训练集和验证集
def split_train_val_data(all_data, train_output, val_output, test_size=0.15):
    # 使用 sklearn 的 train_test_split 随机划分数据
    train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=42)

    # 打印训练集和验证集的条数
    print(f"训练数据条数: {len(train_data)}")
    print(f"验证数据条数: {len(val_data)}")

    # 保存训练集和验证集为 .txt 文件
    save_as_txt(train_data, train_output)
    save_as_txt(val_data, val_output)

    print(f"训练集已保存到 {train_output}，验证集已保存到 {val_output}")

# 主函数：合并文件并生成训练集和验证集
def process_and_split_json_files(input_folder, output_folder, test_size=0.15):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 合并所有 JSON 文件为一个数据集
    all_data = merge_all_json_files(input_folder)

    # 定义训练集和验证集的输出路径
    train_output = os.path.join(output_folder, "merged_train.txt")
    val_output = os.path.join(output_folder, "merged_val.txt")

    # 随机划分并保存训练集和验证集
    split_train_val_data(all_data, train_output, val_output, test_size)

# 示例：处理文件夹中的所有 JSON 文件，合并后划分为训练集和验证集
input_folder = './processed_data'  # 已处理 JSON 文件的文件夹
output_folder = './split_data'  # 保存训练集和验证集的文件夹

# 处理所有 .json 文件并进行数据集合并和划分
process_and_split_json_files(input_folder, output_folder)