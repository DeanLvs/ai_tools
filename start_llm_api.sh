#!/bin/bash
# 放到 /nvme0n1-disk/ai_tools/start_llm_api.sh

echo "[INFO] Restarting llm API server..."

# 杀死原来的进程
pkill -f "uvicorn .*llmLoraAPI:app"

# 启动新进程（你可以换成 supervisor、tmux、pm2 等）
nohup uvicorn llmLoraAPI:app --host 0.0.0.0 --port 6872 > /nvme0n1-disk/book_yes/logs/llm_api.log 2>&1 &
echo "[INFO] LLM API started on port 6872"