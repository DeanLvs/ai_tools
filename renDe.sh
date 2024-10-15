#!/bin/bash

# 获取进程ID
pid1=$(ps -ef | grep '/usr/local/bin/python3.10 dense_pose_api.py' | grep -v grep | awk '{print $2}')
pid2=$(ps -ef | grep 'python3.10 -u dense_pose_api.py' | grep -v grep | awk '{print $2}')

# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for '/usr/local/bin/python3.10 webapp4.0.py'."
fi

if [ -n "$pid2" ]; then
  kill -9 $pid2
  echo "Process $pid2 killed."
else
  echo "No matching process found for 'python3.10 -u webapp4.0.py'."
fi


# 重启 webap
nohup python3.10 -u dense_pose_api.py > xqpq.log 2>&1 &
echo "Webapp started."
