# 获取进程ID
# 使用绝对路径加载 conda 环境
pid1=$(ps -ef | grep 'python3.10 -u /nvme0n1-disk/ai_tools/SocAIty/face2face/API.py' | grep -v grep | awk '{print $2}')

# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for 'API.py'."
fi

nohup python3.10 -u /nvme0n1-disk/ai_tools/SocAIty/face2face/API.py > face_swap_faces.log 2>&1 &
echo "Webapp started."
