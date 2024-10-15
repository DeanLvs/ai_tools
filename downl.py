# -*- coding: utf-8 -*-
import oss2
import os

# 使用环境变量获取认证信息
os.environ['OSS_ACCESS_KEY_ID'] = 'LTAI5tLQFvEjUY4RVCPFpM5T'
os.environ['OSS_ACCESS_KEY_SECRET'] = 'CmkBRQlyjlTpP9FobqmIn1XOLTFhkB'

from oss2.credentials import EnvironmentVariableCredentialsProvider
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

# 初始化 Bucket 实例
bucket = oss2.Bucket(auth, 'https://oss-us-east-1.aliyuncs.com', 'engpusys')

# 下载 pytorch3d_env_backup.yaml 文件
oss2.resumable_download(bucket,'pytorch3d_env_backup.yaml', './pytorch3d_env_backup.yaml')

# 下载 simswap_env_backup.yaml 文件
oss2.resumable_download(bucket,'simswap_env_backup.yaml', './simswap_env_backup.yaml')

# 下载 requirements.txt 文件
oss2.resumable_download(bucket,'requirements.txt', './requirements.txt')

# 下载 transBody_backup.tar.gz 文件
oss2.resumable_download(bucket,'transBody_backup.tar.gz', './transBody_backup.tar.gz')

# 下载 swapface_backup.tar.gz 文件
oss2.resumable_download(bucket,'swapface_backup.tar.gz', './swapface_backup.tar.gz')

# 下载 ai_tools_backup.tar.gz 文件
oss2.resumable_download(bucket,'cai_tools_backup.tar.gz', './ai_tools_backup.tar.gz')

print("文件下载完成！")