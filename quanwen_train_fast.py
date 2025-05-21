# train_qwen3b_lora_no_eval.py
# Python 3.11-compatible QLoRA 微调脚本（纯训练，无验证）

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
import os
from pathlib import Path

# ----------------- 日志设置 -----------------
import logging
from transformers import logging as hf_logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
hf_logging.set_verbosity_info()

# ----------------- 自定义回调：记录 loss/lr/grad_norm 到文件 -----------------
class LossPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[step {state.global_step:>6}] loss = {logs['loss']:.4f} lr = {logs.get('learning_rate', 0):.2e}")

class LossToFile(TrainerCallback):
    def __init__(self, path="qwen_train_loss.log"):
        self.f = open(path, "a", buffering=1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.f.write(f"{state.global_step}\t{logs['loss']:.6f}\t{logs.get('learning_rate', 0):.3e}\t{logs.get('grad_norm', 0):.4f}\n")

    def on_train_end(self, *args, **kwargs):
        self.f.close()

# ----------------- 0. 加速设置 -----------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["RWKV_JIT_ON"] = "1"

# ----------------- 1. 加载数据 -----------------
dataset_path = "/nvme0n1-disk/ai_tools/wuxia_style_text.jsonl"
ds = load_dataset("json", data_files=dataset_path, split="train")

# ----------------- 2. 模型与 tokenizer 加载 -----------------
base_model_path = "/nvme0n1-disk/llm/axolotl/models/Josiefied-Qwen3-8B"
tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_cfg,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_cfg = LoraConfig(
    r=16, lora_alpha=48, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_cfg)

# ----------------- 3. tokenizer 处理（不做人工分词） -----------------
def tok_fn(ex):
    return tok(ex["text"], max_length=512, truncation=True, padding="max_length")

ds_tok = ds.map(tok_fn, batched=True, num_proc=4, remove_columns=["text"]).with_format("torch")

# ----------------- 4. 训练参数配置 -----------------
output_dir = "./output_qwen3_noeval"
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_steps=2000,
    max_steps=50000,
    fp16=False,
    bf16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    save_steps=1000,
    save_total_limit=3,
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True
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
    ckpt_dir = Path(output_dir)
    latest = None
    for sub in ckpt_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("checkpoint-"):
            step = int(sub.name.split("-")[-1])
            latest = sub if latest is None or step > int(latest.name.split("-")[-1]) else latest

    trainer.train(resume_from_checkpoint=str(latest) if latest else None)
