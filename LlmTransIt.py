from datasets import load_dataset
import json
from collections import defaultdict

#加载 Haruhi 数据集
print("📦 Loading dataset...")
dataset = load_dataset("silk-road/ChatHaruhi-54K-Role-Playing-Dialogue")['train']

#提取前 N 个角色，构造角色ID映射（用于枚举优化）
print("🔍 Building character ID mapping...")
role_counter = defaultdict(int)
for sample in dataset:
    character = sample.get("character", "未知角色")
    role_counter[character] += 1

# 按角色频率排序，分配角色ID：<RL_01>, <RL_02>, ...
sorted_roles = sorted(role_counter.items(), key=lambda x: -x[1])
role_map = {
    char: f"<RL_{i+1:02d}>" for i, (char, _) in enumerate(sorted_roles[:50])  # 可根据需要调整数量
}
print("Top characters mapped:")
for char, rid in list(role_map.items())[:5]:
    print(f"  {rid}: {char}")

#开始转换样本
output_path = "chat_haruhi_structured.jsonl"
written = 0

print("🛠 Converting samples...")
with open(output_path, "w", encoding="utf-8") as f_out:
    for sample in dataset:
        character = sample.get("character", "未知角色")
        role_id = role_map.get(character, "<RL_OTHER>")
        convs = sample.get("conversation", [])

        for i in range(len(convs) - 1):
            user = convs[i]
            bot = convs[i + 1]
            if user["from"] == "human" and bot["from"] == "gpt":
                prompt = f"<ROLE> {role_id}\n<ACT> \n<DIA> {user['value']}"
                completion = f"<PSY> \n<ACT> \n<DIA> {bot['value']}"
                f_out.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
                written += 1

print(f"Done! {written} samples written to {output_path}")

#输出角色映射文件供后续使用
dict_path = "character_role_map.json"
with open(dict_path, "w", encoding="utf-8") as f:
    json.dump(role_map, f, ensure_ascii=False, indent=2)
print(f"Saved role ID mapping to {dict_path}")
