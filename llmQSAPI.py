# chat_server.py  (multi-character + user_role edition)
import sqlite3, threading, asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from peft import PeftModel
from book_yes_logger_config import logger
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
    TextGenerationPipeline,StoppingCriteria, StoppingCriteriaList
)
from fastapi import FastAPI, BackgroundTasks
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn, os

PLACE_U = "<USER>"
PLACE_C = "<CHAR>"

# ------------------------------------------------------------------ #
# 0. 全局配置
# ------------------------------------------------------------------ #
# MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/Yi-6B-Chat")
# LORA_DIR = "/nvme0n1-disk/llm/axolotl/models/lora/Yi-6B-Chat-spicy-lora"
LORA_DIR = "/nvme0n1-disk/ai_tools/output_qlora_maxxx_yi9b/checkpoint-20000"
MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/Yi-9B")
# MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/mistral-7b-zh")

# MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/Josiefied-Qwen3-8B")
# LORA_DIR = "/nvme0n1-disk/ai_tools/output_qwen3_noeval/checkpoint-7000"
MAX_TOKEN_BUDGET = 4096
RESERVE_FOR_GEN  = 128
MIN_FOR_GEN  = 64
DB_NAME = "chat_history_v2"

# ---------------------- FastAPI 初始化 ---------------------- #
app = FastAPI()
tokenizer = None
model = None

# 可调参数，建议在文本生成时设置为较高值（温度不要太高）
TOP_P = 0.9  # Top-p (nucleus sampling)，范围0到1
TOP_K = 80  # Top-k 采样的K值
TEMPERATURE = 0.3  # 温度参数，控制生成文本的随机性

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global tokenizer, model
    logger.info("Loading model …")

    # 获取当前脚本目录，亦可改为绝对路径
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        current_directory,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(current_directory)

# ------------------------------------------------------------------ #
# 2. 加载模型
# ------------------------------------------------------------------ #

START_SCRIPT_PATH = "/nvme0n1-disk/ai_tools/start_llm_api.sh"

async def restart_program():
    """
    调用 start_sadtalker_api.sh 来重启服务
    """
    print("Restarting via shell script:", START_SCRIPT_PATH)
    exit_code = os.system(f"bash {START_SCRIPT_PATH}")
    print(f"Restart script exit code: {exit_code}")

# ------------------------------------------------------------------ #
# 3. 裁剪历史工具
# ------------------------------------------------------------------ #
def trim_messages(msgs: List[dict]) -> List[dict]:
    while True:
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )
        if len(ids) + RESERVE_FOR_GEN <= MAX_TOKEN_BUDGET:
            return msgs
        if len(msgs) > 2:
            msgs.pop(1)
        else:
            # 仅剩 system + current user 时，截断 user 文本
            content = msgs[-1]["content"]
            truncated_ids = tokenizer(content, add_special_tokens=False).input_ids[: MAX_TOKEN_BUDGET - RESERVE_FOR_GEN]
            msgs[-1]["content"] = tokenizer.decode(truncated_ids)
            return msgs
def trim_messages_no_chat(msgs: List[str]) -> List[str]:
    """
    给定多行对话，拼成一段文本，再按 token 预算裁剪：
    - 如果整体不超预算就原样返回
    - 超出预算就从最前面开始丢行
    - 如果只剩一行，还超长，就截断这一行
    """
    while True:
        text = "\n".join(msgs)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) + RESERVE_FOR_GEN <= MAX_TOKEN_BUDGET:
            return msgs
        if len(msgs) > 1:
            msgs.pop(0)
        else:
            # 只有一条时，截断这一行
            cut = ids[: MAX_TOKEN_BUDGET - RESERVE_FOR_GEN]
            msgs[0] = tokenizer.decode(cut, skip_special_tokens=True)
            return msgs
class Req(BaseModel):
    char_id: str
    user_id: str
    char_time: str = Field("这天晚上放学后，学生都离开了", description="场景")
    user_rel: str = Field("学生", description="用户角色")
    user_name: str = Field("李石", description="用户姓名")
    user_sex: str = Field("男", description="用户性别")
    char_name: str = Field("赵甜", description="角色姓名")
    char_sex: str = Field("女", description="角色性别")
    char_age: str = Field("36岁", description="角色年龄")
    char_zhiye: str = Field("高中语文老师", description="角色设定")
    char_changjing: str = Field("在教室收拾卫生", description="场景")
    char_des: str = Field(..., description="角色设定")
    user_role: str = Field(..., description="你在对话中扮演的角色，例如“王国的勇士”")
    message: str
    history_turns: int = Field(20, description="取最近 N 轮问答")
    history_external: Optional[List[Dict[str, str]]] = None
class Resp(BaseModel):
    answer: str
    history: List[dict]

def build_prompt_no_chat_v(r: BaseModel) -> str:
    persona = (
        f"{r.char_time} {r.char_name}{r.char_changjing} "
        f"这时, {r.user_name} 走了进来。\n"
    )
    system = (
        "成人小说续写，注意生成内容要保持上下文的人物性格和设定\n"
        f"{r.char_name}（{r.char_sex}）今年{r.char_age} 是{r.char_zhiye} {r.char_des} \n"
        f"{r.user_name}（{r.user_sex}）{r.user_role}\n"
        "严禁出现第三个人名。\n"
        "必要时可描增加身材与着装描写\n"
        "主动推动剧情\n"
        f"{persona}"
    )
    # 拿历史，拼成 ["用户：...","助手：..."] 列表
    past = []
    if r.history_external:
        past = r.history_external
    lines = [system]
    for msg in past:
        role = r.user_name if msg["role"] == "user" else r.char_name
        lines.append(f"{role}：{msg['content']}")
    lines.append(f"{r.user_name}：{r.message}")
    lines.append(f"{r.char_name}：")

    lines = trim_messages_no_chat(lines)
    return "\n".join(lines)

@app.get("/re")
async def re_start_program(tasks: BackgroundTasks):
    """
    提供一个 GET 接口触发后台重启
    """
    print("Preparing to restart service via shell script...")
    tasks.add_task(restart_program)
    return JSONResponse(content={"status": "restarting via shell script"})
@app.post("/chat_no_chat", response_model=Resp)
def chat_no_chat(r: Req):
    # 系统指令（建议为空）
    messages = [
        {"role": "system", "content": ""}
    ]

    if tokenizer is None:
        load_model()
    try:
        logger.info(f"had receive r \n{r}")
        prompt = build_prompt_no_chat_v(r)
        logger.info(f"[Prompt 输入文本] >>>\n{prompt}\n<<<")
        # 获取用户输入
        user_input = input("User: ").strip()
        # 添加用户输入到对话
        messages.append({"role": "user", "content": user_input})

        # 准备输入文本
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # 生成响应
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # 避免警告
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # 解码并打印响应
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Assistant: {response}")
    except Exception as e:
        logger.error("LLM error: %s", e)
        raise HTTPException(500, "LLM failed")

    log = {
        "answer": response
    }
    logger.info(log)
    return log

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6872, workers=1)