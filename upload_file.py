# -*- coding: utf-8 -*-
import oss2
import os

os.environ['OSS_ACCESS_KEY_ID'] = 'LTAI5tLQFvEjUY4RVCPFpM5T'
os.environ['OSS_ACCESS_KEY_SECRET'] = 'CmkBRQlyjlTpP9FobqmIn1XOLTFhkB'
from oss2.credentials import EnvironmentVariableCredentialsProvider
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, 'https://oss-cn-wulanchabu-internal.aliyuncs.com', 'gpusys')
oss2.resumable_upload(bucket, 'pytorch3d_env_backup.yaml', '/mnt/sessd/pytorch3d_env_backup.yaml')
oss2.resumable_upload(bucket, 'simswap_env_backup.yaml', '/mnt/sessd/simswap_env_backup.yaml')
oss2.resumable_upload(bucket, 'requirements.txt', '/mnt/sessd/simswap_env_backup.yaml')
