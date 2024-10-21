#!/bin/bash

# 激活虚拟环境
source /nvme0n1-disk/flux/xflux_env/bin/activate

# 获取正在运行的 python 进程的 PID
pid1=$(ps -ef | grep 'python3 -u /nvme0n1-disk/flux/x-flux/mainApi.py' | grep -v grep | awk '{print $2}')

# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for 'mainApi.py'."
fi

# 启动新的进程
nohup python3 -u /nvme0n1-disk/flux/x-flux/mainApi.py > /nvme0n1-disk/flux/x-flux/flux_ip_gen_imgs.log 2>&1 &
echo "Webapp started."