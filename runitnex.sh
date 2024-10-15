#!/bin/bash

# 设置 CivitAI API token
CIVITAI_API_TOKEN="6342a8596b95d4899ca749fb8f044928"

# 后台运行下载命令
nohup python download.py "https://civitai.com/api/download/models/697868" . --api-token $CIVITAI_API_TOKEN &
nohup python download.py "https://civitai.com/api/download/models/699905" . --api-token $CIVITAI_API_TOKEN &


