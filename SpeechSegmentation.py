from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection


class SpeechSegmentation:
    def __init__(self, access_token):
        self.access_token = access_token
        self.pipeline_talk = None
        self.pipeline_detection = None
        self.HYPER_PARAMETERS = {
            "min_duration_on": 0.0,
            "min_duration_off": 0.0
        }

    def load_model(self):
        self.pipeline_talk = VoiceActivityDetection(segmentation="pyannote/segmentation-3.0",
                                                    use_auth_token=self.access_token)
        self.pipeline_talk.instantiate(self.HYPER_PARAMETERS)
        self.pipeline_detection = OverlappedSpeechDetection(segmentation="pyannote/segmentation-3.0",
                                                            use_auth_token=self.access_token)
        self.pipeline_detection.instantiate(self.HYPER_PARAMETERS)

    def unload_model(self):
        self.pipeline_talk = None
        self.pipeline_detection = None

    def segment(self, audio_file):
        if not self.pipeline_talk or not self.pipeline_detection:
            raise Exception("Model is not loaded. Please load the model first.")

        vad = self.pipeline_talk(audio_file)
        osd = self.pipeline_detection(audio_file)
        return vad, osd


# 使用示例
if __name__ == "__main__":
    access_token = "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"
    audio_file = "/Users/dean/Downloads/live.wav"

    segmenter = SpeechSegmentation(access_token)
    segmenter.load_model()  # 加载模型
    vad, osd = segmenter.segment(audio_file)
    segmenter.unload_model()  # 卸载模型

    print(vad)
    print(osd)