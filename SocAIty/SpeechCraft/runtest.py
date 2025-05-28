from speechcraft import text2voice, voice2embedding, voice2voice

# simple text2speech synthesis
text = "I love society [laughs]! [happy] What a day to make voice overs with artificial intelligence."
audio_numpy, sample_rate = text2voice(text, voice="en_speaker_3")

# speaker embedding generation
embedding = voice2embedding(audio_file="voice_sample_15s.wav", voice_name="hermine").save_to_speaker_lib()
