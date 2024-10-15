def transform_line(line):

    # 删除 "first_stage_model."
    if line.startswith("first_stage_model."):
        line = line[len("first_stage_model."):]
    else:
        return line
    if "decoder.up.0.block.0.nin_shortcut." in line:
        line = line.replace("decoder.up.0.block.0.nin_shortcut.", "decoder.up_blocks.2.resnets.0.conv_shortcut.", 1)
        return line

    if "decoder.up.1.block.0.nin_shortcut." in line:
        line = line.replace("decoder.up.1.block.0.nin_shortcut.", "decoder.up_blocks.3.resnets.0.conv_shortcut.", 1)
        return line

    # 替换 "encoder.mid." 为 "encoder.mid_block."
    if line.startswith("encoder.mid."):
        line = line.replace("encoder.mid.", "encoder.mid_block.", 1)

    # 替换 "decoder.mid." 为 "decoder.mid_block."
    elif line.startswith("decoder.mid."):
        line = line.replace("decoder.mid.", "decoder.mid_block.", 1)

    # 替换 ".up.N.block.M." 为 ".up_blocks.N.resnets.M."
    if ".up." in line and ".block." in line:
        parts = line.split(".")
        up_index = parts[2]
        block_index = parts[4]
        line = line.replace(f".up.{up_index}.block.{block_index}.", f".up_blocks.{up_index}.resnets.{block_index}.")

    # 替换 ".down.N.block.M." 为 ".down_blocks.N.resnets.M."
    if ".down." in line and ".block." in line:
        parts = line.split(".")
        down_index = parts[2]
        block_index = parts[4]
        line = line.replace(f".down.{down_index}.block.{block_index}.",
                            f".down_blocks.{down_index}.resnets.{block_index}.")

    # 替换 "attn_" 为 "attentions."
    if "attn_" in line:
        parts = line.split("attn_")
        index = int(parts[1][0]) - 1  # 减 1
        line = parts[0] + f"attentions.{index}" + parts[1][1:]

    # 替换 "block_" 为 "resnets."
    elif "block_" in line:
        parts = line.split("block_")
        index = int(parts[1][0]) - 1  # 减 1
        line = parts[0] + f"resnets.{index}" + parts[1][1:]

    if "decoder.up." in line and ".upsample." in line:
        parts = line.split(".")
        up_index = int(parts[2]) - 1  # 获取并减1
        line = line.replace(f"decoder.up.{parts[2]}.upsample.", f"decoder.up_blocks.{up_index}.upsamplers.0.")
        return line

    if "encoder.down." in line and ".downsample." in line:
        parts = line.split(".")
        up_index = int(parts[2])
        line = line.replace(f"encoder.down.{parts[2]}.downsample.", f"encoder.down_blocks.{up_index}.downsamplers.0.")
        return line

    # 替换 ".k." 为 ".to_k."
    line = line.replace(".k.", ".to_k.")
    line = line.replace(".q.", ".to_q.")
    line = line.replace(".v.", ".to_v.")

    # 替换 ".norm." 为 ".group_norm."
    line = line.replace(".norm.", ".group_norm.")
    line = line.replace(".norm_out.", ".conv_norm_out.")

    # 替换 ".proj_out." 为 ".to_out.0."
    line = line.replace(".proj_out.", ".to_out.0.")
    line = line.replace(".nin_shortcut.", ".conv_shortcut.")

    return line
