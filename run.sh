#!/bin/bash
sh renDe.sh
sh runLama.sh
sh open_opse.sh
sh depth_es.sh
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

# 备份文件
src_dir="static/uploads/"
backup_dir="static/uploads_back/"

# 创建备份目录（如果不存在）
mkdir -p $backup_dir

# 备份文件，防止重名
for file in $src_dir*; do
  if [ -f "$file" ]; then
    base_name=$(basename "$file")
    backup_file="$backup_dir$base_name"
    
    # 检查文件是否已存在
    if [ -e "$backup_file" ]; then
      # 文件已存在，生成新的文件名
      timestamp=$(date +"%Y%m%d%H%M%S")
      backup_file="${backup_dir}${base_name}_${timestamp}"
    fi
    
    # 复制文件
    cp "$file" "$backup_file"
    echo "File $file backed up as $backup_file"
  fi
done

# 清空并重新创建 uploads 目录
rm -rf $src_dir
mkdir -p $src_dir

# 重启 webap
nohup python3.10 -u webapp9.0.py &
echo "Webapp started."
