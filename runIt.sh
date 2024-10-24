#!/bin/bash
sh renDe.sh
sh runLama.sh
sh open_opse.sh
sh depth_es.sh
sh cloths_mask_api.sh
sh ip_text_gen_img.sh
# conda activate pytorch3d_env
sh /nvme0n1-disk/transBody/PIXIE/run.sh
# source /nvme0n1-disk/flux/xflux_env/bin/activate
sh /nvme0n1-disk/flux/x-flux/start.sh
# 获取进程ID
pid1=$(ps -ef | grep '/usr/local/bin/python3.10 webapp9.0.py' | grep -v grep | awk '{print $2}')
pid2=$(ps -ef | grep 'python3.10 -u webapp9.0.py' | grep -v grep | awk '{print $2}')

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
nohup python3.10 -u webapp9.0.py &

# conda activate simswap_env
cd /nvme0n1-disk/swapface/SimSwap/
sh start.sh

#source /nvme0n1-disk/miniconda3/etc/profile.d/conda.sh
#conda activate ghost
cd /nvme0n1-disk/ghost/sber-swap/
sh start.sh
cd /nvme0n1-disk/ghost/sber-swap/
sh start.sh
cd /nvme0n1-disk/ai_tools/SocAIty/face2face
sh start.sh
echo "Webapp started."
