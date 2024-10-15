import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
access_token = "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"
# 加载预训练的处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_auth_token=access_token)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto").to("cuda")

# 下载并打开图像
raw_image = Image.open("/mnt/sessd/ai_tools/static/uploads/IMG_4901.PNG").convert('RGB')

# 生成描述的函数
def ask_question(question, image):
    inputs = processor(image, question, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# 问题列表，用于生成不同的姿势描述
questions = [
    "Is the person standing?",
    "Is the person sitting?",
    "Is the person facing forward?",
    "Is the person facing sideways?"
]

# 生成并打印回答
for question in questions:
    answer = ask_question(question, raw_image)
    print(f"Question: {question}\nAnswer: {answer}\n")