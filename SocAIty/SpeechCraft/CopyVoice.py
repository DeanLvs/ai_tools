from speechcraft import text2voice, voice2embedding, voice2voice
from media_toolkit import AudioFile
# simple text2speech synthesis
text = "我爱中国[laughs]![happy]多好的国家哈哈哈哈，吕成鑫好帅."
audio_numpy, sample_rate = text2voice(text, voice="zh_speaker_9")

audio = AudioFile().from_np_array(audio_numpy, sr=sample_rate)
audio.save("my_new_audio.wav")

# embedding = voice2embedding(audio_file="voice_sample_15s.wav", voice_name="hermine").save_to_speaker_lib()
# audio_with_cloned_voice, audio_sample_rate = text2voice("我爱美国 [laughs]! [happy] 多好的国家哈哈哈哈，吕成鑫好帅.", voice=embedding)  # also works with voice="hermine"
#
# audio_2 = AudioFile().from_np_array(audio_with_cloned_voice, sr=audio_sample_rate)
# audio_2.save("my_new_2_audio.wav")

# cloned_audio, cloned_rate = voice2voice(audio_file="my_audio_file.wav", voice_name="hermine")
# audio_clone = AudioFile().from_np_array(cloned_audio, sr=cloned_rate)
# audio_clone.save("my_new_audio.wav")
