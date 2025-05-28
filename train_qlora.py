# continue_pretrain_yi9b_fa2_fast.py
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch, bitsandbytes as bnb
import os
from pathlib import Path

import logging
from transformers import logging as hf_log

class LossPrinter(TrainerCallback):
    """
    只要 Trainer 触发 on_log，就把 loss 显式打印出来
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step   = state.global_step
            loss   = logs["loss"]
            lr     = logs.get("learning_rate", "n/a")
            print(f"[step {step:>6}]  loss = {loss:6.4f}   lr = {lr}")

class LossToFile(TrainerCallback):
    """
    每次 on_log 时把 4 个关键字段写到文本文件，便于 Excel / 画图分析
        step <tab> loss <tab> lr <tab> grad_norm
    """
    def __init__(self, path: str = "train_loss.log"):
        self.f = open(path, "a", buffering=1)     # 行缓冲 - 实时刷盘

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        self.f.write(
            f"{state.global_step}\t"
            f"{logs['loss']:.6f}\t"
            f"{logs.get('learning_rate', 0):.3e}\t"
            f"{logs.get('grad_norm', 0):.4f}\n"
        )

    def on_train_end(self, *_, **__):             # 训练结束记得关文件
        self.f.close()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,       # 全局 INFO
)
hf_log.set_verbosity_info()   # 让 Trainer 也用 INFO

# ---------- 0. 环境加速开关 ----------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True
os.environ["RWKV_JIT_ON"] = "1"          # 如果装了 triton 2.1+ 也有帮助
OUTPUT_DIR="./output_qlora_maxxx_yi9b"
# ---------- 1. dataset ----------
ds = load_dataset(
    "json",
    data_files="/nvme0n1-disk/ai_tools/wuxia_style_text.jsonl",
    split="train"
)

# ---------- 2. model ----------
BASE = "/nvme0n1-disk/llm/axolotl/models/Yi-9B"
tok  = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=quant_cfg,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)

# 如果关闭 gradient-ckpt 省 10-15% 时间；显存够用（24 GB ≈ 16 GB 占用） 目前是开启
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA（不变）
lora_cfg = LoraConfig(
    r=16, lora_alpha=48, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_cfg)

# Torch 2.0+ 编译
# model = torch.compile(model, mode="reduce-overhead")   # 纯推导，训练也适用


# ---------- 3. tokenize ----------
def tok_fn(ex):
    return tok(
        ex["text"],
        max_length=512, truncation=True, padding="max_length"
    )

ds_tok = (
    ds.map(tok_fn, batched=True, num_proc=4, remove_columns=["text"])
      .shuffle(seed=42)
      .with_format("torch")                       # ⚠️ 避免 Dataset → Tensor 开销
)

# ---------- 4. train args ----------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=14,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=1_000,
    max_steps=50_000,
    fp16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    disable_tqdm=False,
    save_steps=1_000,
    save_total_limit=4,
    logging_steps=50,  # 或者 25
    logging_first_step=True,  # 立刻看到首条日志
    group_by_length=True,
    dataloader_num_workers=6,        # ← 充分利用 CPU 预取
    dataloader_pin_memory=True,      # ← host→GPU 传输更快
    report_to="none",
)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    data_collator=collator,
    callbacks=[LossPrinter(), LossToFile()]
)

if __name__ == "__main__":
    ckpt_dir = Path(OUTPUT_DIR)
    latest = None
    for sub in ckpt_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("checkpoint-"):
            step = int(sub.name.split("-")[-1])
            latest = sub if latest is None or step > int(latest.name.split("-")[-1]) else latest

    trainer.train(resume_from_checkpoint=str(latest) if latest else None)