from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import torch
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("/nvme0n1-disk/bigLLM/chatglm2-6b", trust_remote_code=True, device='cuda')
CHECKPOINT_PATH = "/nvme0n1-disk/bigLLM/chatglm2/checkpoint/checkpoint-3000/"
config = AutoConfig.from_pretrained("/nvme0n1-disk/bigLLM/chatglm2-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("/nvme0n1-disk/bigLLM/chatglm2-6b", config=config, trust_remote_code=True, device='cuda')
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"), map_location="cuda" if torch.cuda.is_available() else "cpu")
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model = model.cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你现在需要扮演一个离异的中年女人，你有一个儿子，在以下场景回答儿子的问题，你只需要输出你作为此角色的回答。场景：你站在儿子面前，裸露这下体，你的手里拿着一条破了的丝袜。儿子问：妈妈，你怎么了？", history=[])
print(response)

