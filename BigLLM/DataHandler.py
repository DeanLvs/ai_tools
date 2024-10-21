import re
import json
import os


# 读取 .txt 文件
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# 文本清洗（去除空格、换行等）
def clean_text(text):
    text = text.strip()  # 去除空格和换行
    text = text.replace('「', '“').replace('」', '”')  # 替换全角引号
    text = re.sub(r'\s+', ' ', text)  # 移除多余的空白符号
    return text


# 提取对话及其后描述（动作、表情、内心活动）
def extract_dialogues_and_actions(text):
    # 匹配对话，并捕获对话后的动作、表情等描述
    pattern = re.compile(r'“(.*?)”\s*([^“”]*)')  # 匹配引号内的对话及其后的描述
    matches = pattern.findall(text)

    dialogues_and_actions = []

    for match in matches:
        dialogue = match[0].strip()  # 对话部分
        action_or_description = match[1].strip()  # 动作或内心活动描述
        combined_text = dialogue  # 组合的文本

        # 如果有动作或描述，则将其组合到对话后面
        if action_or_description:
            combined_text += f" {action_or_description}"

        dialogues_and_actions.append(combined_text)

    return dialogues_and_actions


# 提取场景描述（对话前的文本部分）
def extract_scenes(text):
    # 将对话和场景分开，提取非对话部分作为场景描述
    scenes = re.split(r'“.*?”', text)
    return [scene.strip() for scene in scenes if scene.strip()]


# 将对话、动作和场景合并为多轮对话数据并添加 history
def format_multi_turn_dialogue(dialogues_and_actions, scenes):
    formatted_data = []
    history = []  # 存储多轮对话的历史记录

    # 取对话和场景列表中较小的长度，避免超出索引
    min_len = min(len(scenes), len(dialogues_and_actions) - 1)

    for i in range(min_len):
        prompt = dialogues_and_actions[i]
        response = dialogues_and_actions[i + 1]

        # 生成新的数据点
        data_point = {
            "prompt": f"场景：{scenes[i]} 对话：{prompt}",
            "response": response,
            "history": history.copy()  # 复制当前的对话历史
        }
        formatted_data.append(data_point)

        # 将当前对话加入历史
        history.append([prompt, response])

        # 只保留最近 3 条对话历史
        if len(history) > 3:
            history.pop(0)  # 移除最早的一条对话，保持 history 长度不超过 3

    return formatted_data


# 保存为 JSON 文件
def save_as_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 处理小说文本，生成带有历史的训练数据
def process_novel_file(file_path, output_path):
    # 读取并清洗文本
    raw_text = read_txt_file(file_path)
    cleaned_text = clean_text(raw_text)

    # 提取对话及动作描述
    dialogues_and_actions = extract_dialogues_and_actions(cleaned_text)

    # 提取场景描述
    scenes = extract_scenes(cleaned_text)

    # 生成带有历史的多轮对话数据
    formatted_data = format_multi_turn_dialogue(dialogues_and_actions, scenes)

    # 保存为 JSON 文件
    save_as_json(formatted_data, output_path)

    print(f"处理完成！训练数据已保存到 {output_path}")


# 处理所有 .txt 文件
def process_all_txt_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.json")
            process_novel_file(file_path, output_path)


# 示例：处理文件夹中的所有小说 .txt 文件，并保存为 .json 格式
input_folder = './transData'  # 小说 .txt 文件所在文件夹
output_folder = './processed_data'  # 处理后保存 JSON 文件的文件夹

# 处理所有 .txt 文件
process_all_txt_files(input_folder, output_folder)

import json
import os
from sklearn.model_selection import train_test_split


# 保存数据为 JSON 文件
def save_as_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 从处理后的数据中随机划分训练集和验证集
def split_train_val_data(input_file, train_output, val_output, test_size=0.2):
    # 读取已处理的 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用 sklearn 的 train_test_split 随机划分数据
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)

    # 保存训练集和验证集
    save_as_json(train_data, train_output)
    save_as_json(val_data, val_output)

    print(f"训练集已保存到 {train_output}，验证集已保存到 {val_output}")


# 主函数：处理文件夹中的所有 .json 文件，生成训练集和验证集
def process_and_split_all_json_files(input_folder, output_folder, test_size=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_folder, file_name)
            train_output = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_train.json")
            val_output = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_val.json")

            # 调用函数，随机划分并保存训练集和验证集
            split_train_val_data(file_path, train_output, val_output, test_size)


# 示例：处理文件夹中的所有 .json 文件，并保存为训练集和验证集
input_folder = './processed_data'  # 已处理 JSON 文件的文件夹
output_folder = './split_data'  # 保存训练集和验证集的文件夹

# 处理所有 .json 文件并进行数据集划分
process_and_split_all_json_files(input_folder, output_folder)

