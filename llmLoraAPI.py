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

PLACE_U = "<U>"
PLACE_C = "<C>"

# ------------------------------------------------------------------ #
# 0. 全局配置
# ------------------------------------------------------------------ #
# MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/Yi-6B-Chat")
# LORA_DIR = "/nvme0n1-disk/llm/axolotl/models/lora/Yi-6B-Chat-spicy-lora"
LORA_DIR = "/nvme0n1-disk/ai_tools/output_qlora_maxxx_yi9b/checkpoint-20000"
MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/Yi-9B")
# MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/Qwen2.5-Sex")
LOAD_LORA = True
# MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/mistral-7b-zh")

# MODEL_DIR = Path("/nvme0n1-disk/llm/axolotl/models/Josiefied-Qwen3-8B")
# LORA_DIR = "/nvme0n1-disk/ai_tools/output_qwen3_noeval/checkpoint-7000"
MAX_TOKEN_BUDGET = 4096
RESERVE_FOR_GEN  = 128
MIN_FOR_GEN  = 64
DB_NAME = "chat_history_v2"

# ---------------------- FastAPI 初始化 ---------------------- #
app = FastAPI()
tok = None
model = None
gen = None
base_model = None
STOP_LIST = None

class KeywordStopper(StoppingCriteria):
    """
    只要最后若干 token 与 keywords 中任意一个序列完全匹配，就停止。
    用法:
        stopper = KeywordStopper(tokenizer, ["\n<USER>：", "\n<CHAR>："])
        model.generate(...,
                       stopping_criteria=StoppingCriteriaList([stopper]))
    """
    def __init__(self, tokenizer, keywords, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        # 预编码成 ids，提高效率
        self.keywords_ids = [
            torch.tensor(tokenizer.encode(k,
                                          add_special_tokens=False),
                         device=self.device)
            for k in keywords
        ]

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids shape: (batch, seq_len)
        for kw_ids in self.keywords_ids:
            if kw_ids.numel() > input_ids.size(1):
                # 关键字比当前序列还长，不可能匹配
                continue
            if torch.equal(input_ids[0, -kw_ids.numel():], kw_ids):
                return True
        return False


def load_model():
    global model, tok, gen, STOP_LIST
    logger.info("Loading model …")
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    # ① 确保占位符是独立 token
    tok.add_special_tokens({'additional_special_tokens': [PLACE_U, PLACE_C]})
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    base_model.resize_token_embeddings(len(tok))
    # 2) 加载去拦截的 spicy LoRA
    if LOAD_LORA:
        model = PeftModel.from_pretrained(base_model, LORA_DIR, local_files_only=True)
    else:
        model = base_model
    # model = PeftModel.from_pretrained(
    #     base_model,
    #     LORA_DIR,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )

    # model = base_model

    gen = TextGenerationPipeline(
        model=model,
        tokenizer=tok,
        return_full_text=False
    )

    STOPER = KeywordStopper(
        tok,
        keywords=[f"\n{PLACE_U}：", f"\n{PLACE_C}："],
        device=model.device  # 避免 CPU↔GPU 拷贝
    )
    STOP_LIST = StoppingCriteriaList([STOPER])


# ------------------------------------------------------------------ #
# 1. 历史管理（含 char_id）
# ------------------------------------------------------------------ #
class ChatHistory:
    _lock = threading.Lock()
    _inst = None
    def __new__(cls, *args, **kwargs):
        if not cls._inst:
            with cls._lock:
                if not cls._inst:
                    cls._inst = super().__new__(cls)
        return cls._inst

    def __init__(self, db: str = DB_NAME + ".db"):
        if getattr(self, "_initd", False):
            return
        self.db = db
        self._setup()
        self._initd = True

    def _conn(self):
        return sqlite3.connect(self.db, check_same_thread=False)

    def _setup(self):
        con = self._conn()
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                char_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
        """)
        con.commit()
        con.close()

    def add(self, user_id: str, char_id: str, role: str, content: str):
        con = self._conn()
        con.execute(
            f"INSERT INTO {DB_NAME}(user_id,char_id,role,content,created_at)"
            " VALUES(?,?,?,?,?)",
            (user_id, char_id, role, content, datetime.utcnow().isoformat())
        )
        con.commit()
        con.close()

    def recent(self, user_id: str, char_id: str, limit: int) -> List[dict]:
        con = self._conn()
        cur = con.execute(
            f"SELECT role,content FROM {DB_NAME}"
            " WHERE user_id=? AND char_id=?"
            " ORDER BY id DESC LIMIT ?",
            (user_id, char_id, limit)
        )
        rows = cur.fetchall()
        con.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

history = ChatHistory()

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
        ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )
        if len(ids) + RESERVE_FOR_GEN <= MAX_TOKEN_BUDGET:
            return msgs
        if len(msgs) > 2:
            msgs.pop(1)
        else:
            # 仅剩 system + current user 时，截断 user 文本
            content = msgs[-1]["content"]
            truncated_ids = tok(content, add_special_tokens=False).input_ids[: MAX_TOKEN_BUDGET - RESERVE_FOR_GEN]
            msgs[-1]["content"] = tok.decode(truncated_ids)
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
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) + RESERVE_FOR_GEN <= MAX_TOKEN_BUDGET:
            return msgs
        if len(msgs) > 1:
            msgs.pop(0)
        else:
            # 只有一条时，截断这一行
            cut = ids[: MAX_TOKEN_BUDGET - RESERVE_FOR_GEN]
            msgs[0] = tok.decode(cut, skip_special_tokens=True)
            return msgs
class Req(BaseModel):
    max_new_tokens: int = Field(RESERVE_FOR_GEN, description="场景"),
    min_new_tokens: int = Field(MIN_FOR_GEN, description="场景"),
    temperature: float = Field(0.9, description="场景"),
    top_p: float = Field(0.9, description="场景"),
    top_k: int = Field(40, description="场景"),
    repetition_penalty: float = Field(1.1, description="场景"),
    no_repeat_ngram_size:int = Field(2, description="场景"),
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

def build_prompt_no_chat(r: BaseModel) -> str:
    persona = (
        f"{r.char_time}{PLACE_C}{r.char_changjing}，{PLACE_C}是{r.char_zhiye}"
        f"，今年{r.char_age}，她{r.char_des}\n"
        f"这时，她的{r.user_rel}{PLACE_U}走了进来。{PLACE_U}{r.user_role}\n"
    )
    system = (
        "成人小说续写，注意内容要符合各自的性别，生成内容要保持上下文的人物性格和设定\n"
        "可适当增加描写身材与着装\n"
        "严禁出现第三个人名。\n"
        "主动推送剧情\n"
        f"输出中只能包含占位符 {PLACE_U} 与 {PLACE_C}，不得出现真实人名，主动推动剧情\n"
        f"{PLACE_U} 的名字是 {r.user_name} 性别是 {r.user_sex}\n"
        f"{PLACE_C} 的名字是 {r.char_name} 性别是 {r.char_sex}\n"
        f"{persona}"
    )
    # 拿历史，拼成 ["用户：...","助手：..."] 列表
    past = []
    if r.history_external:
        past = r.history_external
    elif r.history_turns > 0:
        past = history.recent(r.user_id, r.char_id, r.history_turns * 2)

    lines = [system]
    for msg in past:
        role = PLACE_U if msg["role"] == "user" else PLACE_C
        lines.append(f"{role}说：{msg['content']}")
    lines.append(f"{PLACE_U}说：{r.message}")
    lines.append(f"{PLACE_C}说：")

    lines = trim_messages_no_chat(lines)
    return "\n".join(lines)


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
    elif r.history_turns > 0:
        past = history.recent(r.user_id, r.char_id, r.history_turns * 2)

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
    if gen is None:
        load_model()
    try:
        logger.info(f"had receive r \n{r}")
        prompt = build_prompt_no_chat(r)
        logger.info(f"[Prompt 输入文本] >>>\n{prompt}\n<<<")
        logger.info(f"config max_new_tokens {r.max_new_tokens} min_new_tokens {r.min_new_tokens} temperature {r.temperature} top_p {r.top_p} top_k {r.top_k} repetition_penalty {r.repetition_penalty} no_repeat_ngram_size {r.no_repeat_ngram_size}")
        out = gen(
            prompt,
            max_new_tokens=r.max_new_tokens,
            min_new_tokens=r.min_new_tokens,
            do_sample=True,
            early_stopping=True,
            temperature=r.temperature,
            top_p=r.top_p,
            top_k=r.top_k,
            repetition_penalty=r.repetition_penalty,
            no_repeat_ngram_size=r.no_repeat_ngram_size,
            # stopping_criteria=STOP_LIST
            # stop=[f"\n{PLACE_U}：", f"\n{PLACE_C}："]
        )[0]["generated_text"]
    except Exception as e:
        logger.error("LLM error: %s", e)
        raise HTTPException(500, "LLM failed")
    # 拆出 Assistant 的回答行
    answer = out.split("\n", 1)[0]
    # 去掉多余前缀（如 “result：”）
    if answer.startswith("result："):
        answer = answer[len("result："):]
    # 存库
    history.add(r.user_id, r.char_id, "user", r.message)
    history.add(r.user_id, r.char_id, "assistant", answer)
    answer = answer.replace(PLACE_U, r.user_name).replace(PLACE_C, r.char_name)
    log = {
        "answer": answer,
        "history": history.recent(r.user_id, r.char_id, r.history_turns * 2) + [{"role":"assistant","content":answer}]
    }
    logger.info(log)
    return log

import json
from transformers import TextIteratorStreamer
from starlette.responses import StreamingResponse  # FastAPI re-export 也行

@app.post("/chat_no_chat_stream")
def chat_no_chat_stream(r: Req):
    if gen is None:           # 首次调用时加载模型
        load_model()

    prompt = build_prompt_no_chat(r)
    logger.info(f"[Prompt 输入文本] >>>\n{prompt}\n<<<")
    # —— 用迭代式 streamer ——
    streamer = TextIteratorStreamer(
        tok, skip_prompt=True, skip_special_tokens=True
    )

    # ---- 后台线程跑 pipeline，直接传 prompt 字符串 ----
    threading.Thread(
        target=gen,  # ← 直接用 pipeline 作为 target
        kwargs=dict(
            text_inputs=prompt,
            max_new_tokens=RESERVE_FOR_GEN,
            min_new_tokens=MIN_FOR_GEN,
            do_sample=True,
            early_stopping=True,
            temperature=0.9,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            stopping_criteria=STOP_LIST,
            streamer=streamer,  # <<<<<< 关键
            # early_stopping 仅 beam-search 用，这里可省
        ),
        daemon=True
    ).start()

    async def event_stream():
        prev_text = ""  # 上一次已发送的完整文本
        for text in streamer:
            if not text:
                continue
            # 只取“新增加”的部分
            delta = text[len(prev_text):]
            prev_text = text

            # 把 SentencePiece 的前缀空格符号 ▁ 替换成真正空格（可选）
            delta = delta.replace("▁", " ")

            yield f"data: {delta}\n\n"

        # —— 生成结束后写历史 & [DONE] ——
        history.add(r.user_id, r.char_id, "user", r.message)
        history.add(r.user_id, r.char_id, "assistant", prev_text)
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6872, workers=1)