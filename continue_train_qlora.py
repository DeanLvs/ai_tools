#!/usr/bin/env python3
# 继续从 checkpoint-7000 训练到 20 000 step
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, TaskType

# ---------- 路径 / 超参 ----------
BASE_MODEL   = "/nvme0n1-disk/llm/axolotl/models/Yi-9B"
DATA_JSONL   = "/nvme0n1-disk/ai_tools/wuxia_style_text.jsonl"
OUT_DIR      = "./output_qlora_al_yi9b"
RESUME_CKPT  = f"{OUT_DIR}/checkpoint-7000"
MAX_STEPS    = 30_000
BATCH        = 4            # OOM 就减半
ACC_STEPS    = 4
LR           = 2e-5

# ---------- 量化配置 ----------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit            = True,
    bnb_4bit_use_double_quant= True,
    bnb_4bit_quant_type     = "nf4",
    bnb_4bit_compute_dtype  = torch.float16
)

# ---------- tokenizer & dataset ----------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token            # ### FIX ### 某些 Yi 缺 pad_token

ds = load_dataset("json", data_files=DATA_JSONL, split="train")

def tok_func(ex):
    return tok(ex["text"], max_length=512, truncation=True,
               padding="max_length")

ds = ds.map(tok_func, batched=True, remove_columns=["text"])

# ---------- base + 载入已有 LoRA (不要再 get_peft_model) ----------
print("▶ load base-4bit …")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg,
    trust_remote_code=True, device_map="auto"
)

print("▶ attach LoRA & 继续训练 …")
# 直接把旧 adapter 加回来，并设定可训练
lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = PeftModel.from_pretrained(
    base, RESUME_CKPT,
    is_trainable=True,   # ### FIX ###
    config=lora_cfg      # 覆盖保持一致
)
print(model.print_trainable_parameters())

# ---------- Trainer ----------
args = TrainingArguments(
    output_dir            = OUT_DIR,
    per_device_train_batch_size = BATCH,
    gradient_accumulation_steps = ACC_STEPS,
    learning_rate         = LR,
    warmup_steps          = 100,
    max_steps             = MAX_STEPS,
    fp16                  = True,
    logging_steps         = 10,
    save_steps            = 5_000,
    save_total_limit      = 3,
    group_by_length       = True,
    logging_first_step    = True,
    remove_unused_columns = False,   # ### FIX ###
    report_to             = "none"
)

data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()             # resume_from_checkpoint 已经隐含