import torch
import re

def convert_keys(conditioner_state_dict):
    converted_state_dict = {}
    pattern = re.compile(r"conditioner.embedders\.1\.model\.transformer\.resblocks\.(\d+)\.attn\.(.*)")

    for key, value in conditioner_state_dict.items():
        # 使用正则表达式匹配并提取层号和子键名
        match = pattern.match(key)
        if match:
            layer_num = match.group(1)
            sub_key = match.group(2)

            # 构建新的 key
            new_key = f"text_model.encoder.layers.{layer_num}.self_attn."

            if sub_key == "in_proj_weight":
                d = value.shape[0] // 3
                converted_state_dict[new_key + "q_proj.weight"] = value[:d, :]
                converted_state_dict[new_key + "k_proj.weight"] = value[d:2*d, :]
                converted_state_dict[new_key + "v_proj.weight"] = value[2*d:, :]
            elif sub_key == "in_proj_bias":
                d = value.shape[0] // 3
                converted_state_dict[new_key + "q_proj.bias"] = value[:d]
                converted_state_dict[new_key + "k_proj.bias"] = value[d:2*d]
                converted_state_dict[new_key + "v_proj.bias"] = value[2*d:]
            elif sub_key == "out_proj.weight":
                converted_state_dict[new_key + "out_proj.weight"] = value
            elif sub_key == "out_proj.bias":
                converted_state_dict[new_key + "out_proj.bias"] = value
            else:
                # 对于不需要特殊处理的情况，直接添加到新的字典中
                new_key = new_key + sub_key
                converted_state_dict[new_key + sub_key] = value
            # print(
            #     f"keys shape: {key} {new_key} checkpoint shape is {conditioner_state_dict[key].shape}\n")
        elif "conditioner.embedders.1.model.text_projection" == key:
            new_key = "text_projection.weight"
            converted_state_dict[new_key] = value
        elif "conditioner.embedders.1.model.token_embedding.weight" == key:
            new_key = "text_model.embeddings.token_embedding.weight"
            converted_state_dict[new_key] = value
        elif "conditioner.embedders.1.model.positional_embedding" == key:
            new_key = "text_model.embeddings.position_embedding.weight"
            converted_state_dict[new_key] = value
        elif "conditioner.embedders.1.model.ln_final.bias" == key:
            new_key = "text_model.final_layer_norm.bias"
            converted_state_dict[new_key] = value
        elif "conditioner.embedders.1.model.ln_final.weight" in key:
            new_key = "text_model.final_layer_norm.weight"
            converted_state_dict[new_key] = value
        elif ".ln_1." in key or ".ln_2." in key or ".c_fc." in key or ".c_proj." in key:
            # 替换特定的子字符串
            new_key = key.replace("conditioner.embedders.1.model.transformer.resblocks.","text_model.encoder.layers.")
            new_key = new_key.replace(".ln_1.", ".layer_norm1.")
            new_key = new_key.replace(".ln_2.", ".layer_norm2.")
            new_key = new_key.replace(".c_fc.", ".fc1.")
            new_key = new_key.replace(".c_proj.", ".fc2.")
            converted_state_dict[new_key] = value
    # for key in converted_state_dict.keys():
    #     print(f"- {key}")
    return converted_state_dict

# # 示例使用
# conditioner_state_dict = {
#     "2222.conditioner.embedders.1.model.transformer.resblocks.2.attn.in_proj_weight": torch.randn(1536, 512),
#     "2222.conditioner.embedders.1.model.transformer.resblocks.2.attn.in_proj_bias": torch.randn(1536),
#     # 可以添加其他条目...
# }
#
# # 转换键名并输出结果
# converted_weights = convert_keys(conditioner_state_dict)
#
# # 打印转换后的键名和权重形状
# for key, value in converted_weights.items():
#     print(f"{key}: {value.shape}")