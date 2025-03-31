#!/bin/bash

# 获取进程ID
pid1=$(ps -ef | grep 'python cloths_mask_api.py' | grep -v grep | awk '{print $2}')
pid2=$(ps -ef | grep 'python -u cloths_mask_api.py' | grep -v grep | awk '{print $2}')

# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for cloths_mask_api"
fi

if [ -n "$pid2" ]; then
  kill -9 $pid2
  echo "Process $pid2 killed."
else
  echo "No matching process found for cloths_mask_api"
fi

# 重启 webap
nohup python -u cloths_mask_api.py > /nvme0n1-disk/book_yes/logs/cloths_mask.log 2>&1 &
echo "Webapp started."
