import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperTranscriber:
    def __init__(self, model_id="openai/whisper-large-v3", chunk_length_s=25, batch_size=16, device=None):
        # 设置设备和数据类型
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu" if device is None else device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size

        # 初始化模型和处理器为 None
        self.model = None
        self.processor = None
        self.pipe = None

    def load_model(self):
        # 加载模型
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # 配置管道，启用分段、批处理和时间戳
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=self.chunk_length_s,  # 分段长度（秒）
            batch_size=self.batch_size,  # 批处理大小
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True,  # 启用句子级别的时间戳
            generate_kwargs={"language": "chinese"}  # 指定翻译成英文
        )

    def unload_model(self):
        # 清理模型和处理器
        del self.pipe
        del self.model
        del self.processor
        self.pipe = None
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()  # 清理 GPU 内存（如果使用 GPU）

    def transcribe(self, file_path):
        if not self.pipe:
            raise RuntimeError("Model is not loaded. Call 'load_model' before transcribing.")

        # 转录音频并返回时间戳
        result = self.pipe(file_path)

        # 输出结果
        transcription = []
        for chunk in result["chunks"]:
            transcription.append({
                "text": chunk["text"],
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1]
            })

        return transcription


# 使用示例
if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    transcriber.load_model()  # 在程序启动时加载模型

    try:
        file_path = "/Users/dean/Downloads/live.wav"
        transcription = transcriber.transcribe(file_path)
        for segment in transcription:
            print(f"Text: {segment['text']}, Start: {segment['start']}, End: {segment['end']}")
    finally:
        transcriber.unload_model()  # 在程序终止时卸载模型