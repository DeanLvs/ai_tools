from transformers import BarkModel, BarkProcessor
import torchaudio
import torch


# 单例类定义
class BarkInferenceSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BarkInferenceSingleton, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.processor = BarkProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_voice_embedding(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != 24000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(waveform)

        max_length = 60 * 24000  # 60秒的音频长度
        if waveform.size(1) > max_length:
            waveform = waveform[:, :max_length]

        inputs = self.processor(waveforms=waveform, sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            speaker_embedding = self.model.extract_speaker_embeddings(**inputs)
        return speaker_embedding

    def generate_speech(self, text, speaker_embedding, duration=None):
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        inputs["speaker_embeddings"] = speaker_embedding.to(self.device)

        generate_kwargs = {}
        if duration:
            tokens_per_second = len(inputs["input_ids"][0]) / duration
            generate_kwargs["tokens_per_second"] = tokens_per_second

        with torch.no_grad():
            speech_output = self.model.generate(**inputs, **generate_kwargs).cpu().numpy()
        return speech_output


if __name__ == "__main__":


    # 循环 text_lime_line 文本，并在 speech_human_line 中查找到对应时间轴的说话人的音频文件作为 audio_file，传入下边模型开始模仿
    text = "你好，我的名字是Suno。[laughs] 我喜欢吃披萨。[sighs] 但是我也喜欢其他的东西，比如玩井字游戏。[pause] 你呢？"
    audio_file = "path_to_audio_file.wav"
    duration = 15  # 以15秒内完成语音生成为例

    bark_inference = BarkInferenceSingleton()
    speaker_embedding = bark_inference.extract_voice_embedding(audio_file)
    speech_output = bark_inference.generate_speech(text, speaker_embedding, duration=duration)

    # 保存输出到文件
    from scipy.io.wavfile import write

    write("bark_output.wav", 24000, speech_output.squeeze())  # 假设采样率为24000Hz

    print("语音生成完成并保存为 bark_output.wav")