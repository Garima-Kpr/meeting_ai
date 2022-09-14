# instantiate pretrained speaker diarization pipeline
from pyannote.audio import Pipeline
from transformers import pipeline
from pydub import AudioSegment 
from pydub.utils import make_chunks
from transformers import BertTokenizer
import pickle
import os

asr = pipeline(
   "automatic-speech-recognition",
    model="facebook/wav2vec2-large-960h-lv60-self",
    feature_extractor="facebook/wav2vec2-large-960h-lv60-self",
)

speaker_segmentation = Pipeline.from_pretrained("pyannote/speaker-diarization")

def segmentation(audio):
    print("start segmentation to text")
    speaker_output = speaker_segmentation(audio, num_speakers=2)
    print("start speech to text")
    text_output = asr(audio,return_timestamps="word")
    print("end speech to text")
    
    chunks = text_output['chunks']

    diarized_output = []
    i = 0
    for turn, _, speaker in speaker_output.itertracks(yield_label=True):
        diarized = ""
        while i < len(chunks) and chunks[i]['timestamp'][1] <= turn.end:
            diarized += chunks[i]['text'].lower() + ' '
            i += 1
        
        if diarized != "":
            line = "{}: ''{}'' \n".format(speaker,diarized)
            diarized_output.append(line)
        print("*")
    return "".join(diarized_output)


def audio_splitter(audio_path):
    myaudio = AudioSegment.from_file(audio_path, "wav") 
    chunk_length_ms = 120*1000 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    file_names = []
    for i, chunk in enumerate(chunks): 
        chunk_name = "_{0}".format(i) 
        # print ("exporting", chunk_name)
        file_name = audio_path+chunk_name+'.wav'
        chunk.export(file_name, format="wav")
        file_names.append(file_name)
    return file_names

def generate_speech(audio_path):
    splitted_files = audio_splitter(audio_path)
    output_speech = []

    for s_file in splitted_files:
        output = segmentation(s_file)
        output_speech.append(output)
        os.remove(s_file)
    return output_speech


def summarizer(output_speech):
    print("")
    print("start of summarization")
    with open('./summarizer_bart_large_cnn_samsum.pkl', 'rb') as file:
        summarizer = pickle.load(file)
    return summarizer(output_speech, max_length=130, min_length=30, do_sample=False)


audio = ["./data/8415189576557ef37bfc0faffcf8675775aed3416881ba556212c05e.wav",
"./data/lex_fridman.wav"]


print(" ")
#print(generate_speech(audio[0]))

output_speech = generate_speech(audio[1])
print(summarizer(output_speech))


