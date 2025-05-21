# 杀死原来的进程
pkill -f "uvicorn .*proxy_server:app"

# 启动新进程（你可以换成 supervisor、tmux、pm2 等）
nohup uvicorn proxy_server:app --host 0.0.0.0 --port 5008 > /nvme0n1-disk/book_yes/logs/llm_proxy_api.log 2>&1 &
echo "[INFO] LLM proxy_server started on port 5008"

sh start_llm_api.sh
cd /nvme0n1-disk/voice/Spark-TTS
bash start_tts_api.sh

cd /nvme0n1-disk/video/SadTalker
bash start_sadtalker_api.sh

