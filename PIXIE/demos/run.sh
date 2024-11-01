#!/bin/bash
# 获取进程ID
#!/bin/bash

# 使用绝对路径加载 conda 环境
source /nvme0n1-disk/miniconda3/etc/profile.d/conda.sh  # 根据你的 Conda 安装路径调整

# 激活 conda 环境
conda activate pytorch3d_env
pid1=$(ps -ef | grep 'python /nvme0n1-disk/transBody/PIXIE/runIt.py' | grep -v grep | awk '{print $2}')
pid2=$(ps -ef | grep 'python -u /nvme0n1-disk/transBody/PIXIE/runIt.py' | grep -v grep | awk '{print $2}')

# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for runIt"
fi

if [ -n "$pid2" ]; then
  kill -9 $pid2
  echo "Process $pid2 killed."
else
  echo "No matching process found for runIt"
fi


# 重启 webap
nohup python -u /nvme0n1-disk/transBody/PIXIE/runIt.py > /nvme0n1-disk/book_yes/logs/PIXIE.log 2>&1 &
echo "Webapp started."
conda deactivate
