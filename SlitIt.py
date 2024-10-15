from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

class SlitIt:
    def __init__(self, model_source="speechbrain/sepformer-wsj02mix", model_dir='pretrained_models/sepformer-wsj02mix',
                 sample_rate=16000):
        self.model_source = model_source
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.model = None

    def load_model(self):
        self.model = separator.from_hparams(source=self.model_source, savedir=self.model_dir)
        self.model.hparams.sample_rate = self.sample_rate

    def unload_model(self):
        self.model = None

    def separate_audio(self, input_path, output_prefix, iterations=1):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        current_input_path = input_path
        for i in range(iterations):
            est_sources = self.model.separate_file(path=current_input_path)
            for j in range(est_sources.shape[2]):
                output_path = "{output_prefix}_iter{i + 1}_source{j + 1}.wav"
                torchaudio.save(output_path, est_sources[:, :, j].detach().cpu(), self.sample_rate)

            # Use the separated source 1 for the next iteration
            current_input_path = "{output_prefix}_iter{i + 1}_source1.wav"

if __name__ == '__main__':
    # 使用示例
    separatorIt = SlitIt(sample_rate=16000)
    separatorIt.load_model()
    separatorIt.separate_audio('/Users/dean/Downloads/Voice_CN/output_audio.wav', 'output', iterations=1)
    separatorIt.unload_model()