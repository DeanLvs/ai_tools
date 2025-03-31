# -*- coding: utf-8 -*-
import oss2
import os

os.environ['OSS_ACCESS_KEY_ID'] = 'LTAI5tLQFvEjUY4RVCPFpM5T'
os.environ['OSS_ACCESS_KEY_SECRET'] = 'CmkBRQlyjlTpP9FobqmIn1XOLTFhkB'
from oss2.credentials import EnvironmentVariableCredentialsProvider
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, 'https://oss-us-east-1.aliyuncs.com', 'gpusys')
oss2.resumable_upload(bucket, 'ai_tools.tar.gz', '/nvme0n1-disk/gpu_back/ai_tools.tar.gz')
oss2.resumable_upload(bucket, 'swapface.tar.gz', '/nvme0n1-disk/gpu_back/swapface.tar.gz')
oss2.resumable_upload(bucket, 'transBody.tar.gz', '/nvme0n1-disk/gpu_back/transBody.tar.gz')
