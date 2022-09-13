# instantiate pretrained speaker diarization pipeline
import pyannote.audio
Pipeline = pyannote.audio.Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# apply pretrained pipeline
diarization = pipeline("/home/garima/Downloads/8415189576557ef37bfc0faffcf8675775aed3416881ba556212c05e.wav")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_A
# start=1.8s stop=3.9s speaker_B
# start=4.2s stop=5.7s speaker_A
# ...