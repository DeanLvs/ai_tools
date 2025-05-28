import time
from transformers import T5EncoderModel

model_name = 'google/t5-v1_1-xxl'
max_retries = 5
for i in range(max_retries):
    try:
        print(f"Downloading model ({i+1}/{max_retries})...")
        model = T5EncoderModel.from_pretrained(model_name, from_tf=True)
        print("下载完成")
        break
    except Exception as e:
        print(f"下载失败：{e}")
        if i < max_retries - 1:
            print("正在重试...")
            time.sleep(10)
        else:
            print(" 达到最大重试次数，下载失败")
