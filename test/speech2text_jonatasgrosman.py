from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

audio_paths = ["/home/garima/Downloads/8415189576557ef37bfc0faffcf8675775aed3416881ba556212c05e.wav"]
#audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)

print(transcriptions)