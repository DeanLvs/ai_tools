source /nvme0n1-disk/miniconda3/etc/profile.d/conda.sh
conda activate ghost
pid1=$(ps -ef | grep 'python -u /nvme0n1-disk/ghost/sber-swap/sber_swap_api.py' | grep -v grep | awk '{print $2}')
# 检查进程ID是否存在并杀掉进程
if [ -n "$pid1" ]; then
  kill -9 $pid1
  echo "Process $pid1 killed."
else
  echo "No matching process found for 'sber_swap_api.py'."
fi
nohup python -u /nvme0n1-disk/ghost/sber-swap/sber_swap_api.py  > log_sber_swap_api.log 2>&1 &
echo "Webapp started."
conda deactivate