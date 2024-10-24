import json

def calculate_lengths(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    prompt_lengths = []
    response_lengths = []
    history_lengths = []

    for line in lines:
        data = json.loads(line)
        prompt_length = len(data['prompt'])
        response_length = len(data['response'])

        # 计算 history 的总长度
        history_length = 0
        if 'history' in data:
            for history_item in data['history']:
                history_length += len(history_item[0]) + len(history_item[1])

        prompt_lengths.append(prompt_length)
        response_lengths.append(response_length)
        history_lengths.append(history_length)

    return prompt_lengths, response_lengths, history_lengths

def print_length_statistics(lengths, name):
    print(f"{name}长度统计:")
    print(f"最小长度: {min(lengths)}")
    print(f"最大长度: {max(lengths)}")
    print(f"平均长度: {sum(lengths)/len(lengths)}")

# 替换为你的训练文件路径
file_path = './split_data/merged_train.txt'

prompt_lengths, response_lengths, history_lengths = calculate_lengths(file_path)

# 输出统计信息
print_length_statistics(prompt_lengths, "Prompt")
print_length_statistics(response_lengths, "Response")
print_length_statistics(history_lengths, "History")