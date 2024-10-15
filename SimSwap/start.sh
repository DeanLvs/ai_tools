# 获取进程ID
# 使用绝对路径加载 conda 环境
source /nvme0n1-disk/miniconda3/etc/profile.d/conda.sh  # 根据你的 Conda 安装路径调整

conda activate simswap_env
pid1=$(ps -ef | grep 'python -u /nvme0n1-disk/swapface/SimSwap/API.py' | grep -v grep | awk '{print $2}')

# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for 'API.py'."
fi

nohup python -u /nvme0n1-disk/swapface/SimSwap/API.py > swap_faces.log 2>&1 &
echo "Webapp started." 
conda deactivate
