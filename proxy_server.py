from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import uuid
import httpx
import asyncio
import json
from typing import Optional
from datetime import datetime
from fastapi.responses import StreamingResponse
from book_yes_logger_config import logger
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="LobeChat Proxy with TTS & SadTalker")
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 或者 ["https://your.domain.com"]
    allow_credentials=True,
    allow_methods=["*"],           # 允许 GET、POST、OPTIONS 等所有方法
    allow_headers=["*"],           # 允许所有头部
)
# === 配置 ===
LLM_RE_URL         = "http://localhost:6872/re"
TTS_URL            = "http://localhost:5018/process_tts"
TTS_RE_URL         = "http://localhost:5018/re"
SADTALKER_URL      = "http://localhost:5720/process_sadtalker"
SADTALKER_RE_URL   = "http://localhost:5720/re"

TTS_SAVE_DIR       = "./tts_output"
VIDEO_SAVE_DIR     = "./video_output"
TTS_ACCESS_PATH    = "/static/tts"
VIDEO_ACCESS_PATH  = "/static/video"

os.makedirs(TTS_SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

app.mount(TTS_ACCESS_PATH, StaticFiles(directory=TTS_SAVE_DIR), name="tts")
app.mount(VIDEO_ACCESS_PATH, StaticFiles(directory=VIDEO_SAVE_DIR), name="video")

LLM_JSON_URL    = "http://localhost:6872/chat_no_chat"         # 非流式
LLM_STREAM_URL  = "http://localhost:6872/chat_no_chat_stream"  # 流式


# === Pydantic 定义 ===
# === Pydantic ===
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


# === 通用调用 + 释放显存 ===
async def call_and_release(post_url: str, release_url: str, payload: dict, save_dir: str, file_ext: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(post_url, json=payload)
            resp.raise_for_status()
            data = resp.content
        filename = f"{uuid.uuid4().hex}.{file_ext}"
        out_path = os.path.join(save_dir, filename)
        with open(out_path, "wb") as f:
            f.write(data)
        asyncio.create_task(httpx.AsyncClient().get(release_url))
        return filename
    except Exception as e:
        app.logger = getattr(app, "logger", print)
        app.logger(f"Error calling {post_url}: {e}")
        return None


async def synthesize_tts(text: str, prompt_speech_path: Optional[str] = None) -> Optional[str]:
    # 使用请求方传入的 prompt_speech_path，否则回退到 assets/ref_voice.wav
    payload = {
        "text": text,
        "prompt_speech_path": prompt_speech_path or "assets/ref_voice.wav"
    }
    fn = await call_and_release(TTS_URL, TTS_RE_URL, payload, TTS_SAVE_DIR, "zip")
    return f"{TTS_ACCESS_PATH}/{fn}" if fn else None


async def synthesize_video(audio_path: str, image_path: Optional[str] = None) -> Optional[str]:
    # 使用请求方传入的 image_path，否则回退到 assets/ref_face.png
    payload = {
        "driven_audio": audio_path,
        "source_image": image_path or "assets/ref_face.png"
    }
    fn = await call_and_release(SADTALKER_URL, SADTALKER_RE_URL, payload, VIDEO_SAVE_DIR, "zip")
    return f"{VIDEO_ACCESS_PATH}/{fn}" if fn else None

# === 聊天主接口 ===
@app.post("/chat/completions")
async def completions(request: Request, chat_req: ChatRequest):

    logger.info(f'receive {chat_req}')
    # === ✨ 特殊指令：系统消息中要求生成 10 字以内标题 ===
    special = next(
        (m for m in chat_req.messages
         if m.role == "system" and "你是一名擅长会话的助理" in m.content),
        None
    )

    def build_chunk(content: str, finish: bool = False):
        """生成符合 OpenAI ChatCompletion 流式规范的字典"""
        if finish:
            delta = {}
            finish_reason = "stop"
        else:
            delta = {"content": content}
            finish_reason = None
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": chat_req.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }

    if special:
        logger.info("发现可直接返回")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if chat_req.stream:  # 若前端要求 SSE
            def fake_stream():
                yield f"data: {json.dumps(build_chunk(now_str), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(fake_stream(),
                                     media_type="text/event-stream")

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

    llm_payload = {
        "char_id": chat_req.char_id or "default-char",
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
        "message": user_msg,
        "history_turns": 0,
        # messages 部分，你可以按需处理；这里直接传整个列表
        "history_external": [m.dict() for m in filtered_messages],
    }


    # --------------------- 非流式 ---------------------
    if not chat_req.stream:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(LLM_JSON_URL, json=llm_payload)
            resp.raise_for_status()
        reply = resp.json()["answer"]
        # reply = "mock数据"
        # （如需 TTS/SadTalker，可在此同步调用，再拼 reply）
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": chat_req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    async def event_stream():

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(LLM_JSON_URL, json=llm_payload)
            resp.raise_for_status()
        reply = resp.json()["answer"]

        # —— 一定要先把 reply 赋给 answer_acc ——
        answer_acc = reply

        # —— 把整段 reply 作为第一个 delta 推送 ——
        yield f"data: {json.dumps(build_chunk(reply), ensure_ascii=False)}\n\n"
        # —— 2) 等待多媒体生成（同步阻塞，SSE 仍保持打开） —— #
        links_text = ""
        tts_url: Optional[str] = None
        if chat_req.prompt_speech_path:
            tts_url = await synthesize_tts(answer_acc, chat_req.prompt_speech_path)
            if tts_url:
                links_text += f"\n\n🎧 [语音播放]({tts_url})"

        if chat_req.image_path:
            audio_input = tts_url or "assets/ref_voice.wav"
            video_url = await synthesize_video(audio_input, chat_req.image_path)
            if video_url:
                links_text += f"\n\n🎬 [查看动画]({video_url})"

        if links_text:
            # 把多媒体链接当作新的 delta 再推一次
            yield f"data: {json.dumps(build_chunk(links_text))}\n\n"
            answer_acc += links_text  # 如需写历史，可用它

        # —— 3) 发送 stop-chunk + [DONE] —— #
        yield f"data: {json.dumps(build_chunk('', finish=True))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")


# === 启动方式 ===
# pip install fastapi uvicorn httpx
# uvicorn proxy_server:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proxy_server:app", host="0.0.0.0", port=5008, workers=1)