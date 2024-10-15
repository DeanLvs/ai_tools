#!/bin/bash
# 获取进程ID
pid1=$(ps -ef | grep '/usr/local/bin/python3.10 SingletonDepthEstimator.py' | grep -v grep | awk '{print $2}')
pid2=$(ps -ef | grep 'python3.10 -u SingletonDepthEstimator.py' | grep -v grep | awk '{print $2}')

# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for SingletonDepthEstimator"
fi

if [ -n "$pid2" ]; then
  kill -9 $pid2
  echo "Process $pid2 killed."
else
  echo "No matching process found for SingletonDepthEstimator"
fi


# 重启 webap
nohup python3.10 -u SingletonDepthEstimator.py > qqpq.log 2>&1 &
echo "Webapp started."
