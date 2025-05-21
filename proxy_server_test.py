import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import asyncio
app = FastAPI(title="LobeChat Proxy with TTS & SadTalker")

from fastapi import FastAPI, Request
import asyncio

app = FastAPI(title="LobeChat Proxy with TTS & SadTalker")
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 或者 ["https://your.domain.com"]
    allow_credentials=True,
    allow_methods=["*"],           # 允许 GET、POST、OPTIONS 等所有方法
    allow_headers=["*"],           # 允许所有头部
)
# === Pydantic 定义 ===
class Message(BaseModel):
    id: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None

    class Config:
        extra = "allow"


class ChatRequest(BaseModel):
    model: Optional[str] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    messages: List[Message] = Field(default_factory=list)

    # 原来扩展字段
    char_id: Optional[str] = None
    prompt_speech_path: Optional[str] = None
    image_path: Optional[str] = None

    # 你新加的所有字段
    user_id: Optional[str] = None
    user_rel: Optional[str] = None
    user_name: Optional[str] = None
    user_sex: Optional[str] = None
    user_role: Optional[str] = None

    char_time: Optional[str] = None
    char_name: Optional[str] = None
    char_sex: Optional[str] = None
    char_age: Optional[str] = None
    char_zhiye: Optional[str] = None
    char_changjing: Optional[str] = None
    char_des: Optional[str] = None

    class Config:
        extra = "allow"

@app.post("/chat/completions")
async def completions(request: Request, chat_req: ChatRequest):

    print(f'receive {chat_req}')
    # === 筛选 messages，剔除不需要的角色 / 首条 user ===
    original_messages = chat_req.messages[:]

    # 删除所有 system 消息
    filtered_messages = [m for m in original_messages if m.role != "system"]

    # 删除最后一条 user 消息（如果存在）
    for i in range(len(filtered_messages) - 1, -1, -1):
        if filtered_messages[i].role == "user":
            del filtered_messages[i]
            break

    # 提取 user_msg（从原始消息中找最后一条 user）
    try:
        user_msg = next(m.content for m in reversed(original_messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(400, "No user message")

    # ---------- 组装发给 LLM 的 payload ----------
    llm_payload = {
        "char_id": chat_req.char_id,
        "user_id": chat_req.user_id,
        "user_rel": chat_req.user_rel,
        "user_name": chat_req.user_name,
        "user_sex": chat_req.user_sex,
        "user_role": chat_req.user_role,
        "char_time": chat_req.char_time,
        "char_name": chat_req.char_name,
        "char_sex": chat_req.char_sex,
        "char_age": chat_req.char_age,
        "char_zhiye": chat_req.char_zhiye,
        "char_changjing": chat_req.char_changjing,
        "char_des": chat_req.char_des,
        "prompt_speech_path": chat_req.prompt_speech_path,
        "image_path": chat_req.image_path,
        # messages 部分，你可以按需处理；这里直接传整个列表
        "messages": [m.dict() for m in chat_req.messages],
    }
    print(f'llm_payload is {llm_payload}')


    reply = "mac数据"

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 1684500000,  # Unix 时间戳示例
        "model": chat_req.model,  # 例如 "custom-local-model"
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "这是来自后端的模拟回复"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proxy_server_test:app", host="0.0.0.0", port=8000, reload=True)