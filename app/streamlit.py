from logging import exception
import turtle import right, width
import streamlit as st
import librosa
import librosa.display
import torch
import tempfile
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import requests
from gtts import gTTS
import datetime
from pyannote.audio import Pipeline
from transformers import pipeline
from pydub import AudioSegment 
from pydub.utils import make_chunks
from transformers import BertTokenizer
import os
import time
import base64
from fpdf import FPDF
import torchaudio
from speechbrain.pretrained import EncoderClassifier

BASEFILE = "audio.wav"
OUTFILE = "audio_output.wav"
PREFIX = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
SAMPLERATE = 16000  
DURATION = 30 # seconds
CHOICE_LIST = {"Record audio":0,"Upload audio(.wav)":1}


@st.cache
def speech_to_text(audio):
    print("function_started")
    asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-large-960h-lv60-self",
    feature_extractor="facebook/wav2vec2-large-960h-lv60-self",
    )   
    speaker_segmentation = Pipeline.from_pretrained("pyannote/speaker-diarization")

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
    transcript = "".join(diarized_output)
    print(transcript)
    print("function_running")
    return transcript

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
        output = speech_to_text(s_file)
        output_speech.append(output)
        os.remove(s_file)
    return output_speech

def summarizer(output_speech):
    print("")
    print("start of summarization")
    summarizer_model = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    return summarizer_model(output_speech, max_length=130, min_length=30, do_sample=False)


if __name__=="__main__":

    st.title("# MEETINOTES")
    st.subheader("AI based survival kit for boring and unproductive meetings")
    st.image("./tired-businessmen-cartoon-transformed-black.jpg", width = 700)
    st.subheader("Meeting details")
    d = st.date_input(
        "Select the date of meeting",
        datetime.date(2022, 9, 15))
    t = st.time_input(
        'Select the time of meeting', datetime.time(8, 00))
    meeting_host = st.text_input('Meeting topic/ host name', '')

    
    

    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            width: 150%;   
            height: 100%;
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-repeat: no-repeat;
            background-size: contain;
            background-size: fixed
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('DALL·E 2022-09-15 15.41.56.png')  
    
    choice = st.radio("Please select from the two options",CHOICE_LIST.keys())
    
    try:
        if CHOICE_LIST[choice] == 0:
            FILENAME  = "_".join([PREFIX, BASEFILE])
            if  st.button('Record'):
                with st.spinner(f'Recording for {DURATION} seconds ....'):
                    mydata = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE,
                        channels=1, blocking=True)
                    sd.wait()
                    sf.write(FILENAME, mydata, SAMPLERATE)
                st.success("Recording completed")

                try:
                    audio_file = open(FILENAME, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')

                    #speech, rate = librosa.load(FILENAME,sr=SAMPLERATE)
                    if st.button('Speech to text'):
                        transcript_res = speech_to_text(mydata)
                        st.write("Audio Transcript : ",transcript_res)

                    if st.button('Get summary'):
                        summary = summarizer(generate_speech(mydata))
                        c = st.container()
                        c.write("Preparing the summary for you")
                        with st.spinner('Wait for it...'):
                            time.sleep(5)
                        st.success('Done!')
                        summary_text = st.text_input("Report Text", summary)
                        export_as_pdf = st.button("Export Report")

                        with st.expander("Check out the summary of your meeting"):
                            st.write(summary_text)




                except:
                    st.write("Please record audio first")
                    
        
        if CHOICE_LIST[choice] == 1:
            
            st.subheader("Upload a audio file in .wav format")
            st.write("Please upload the meeting audio")
            audio_upload = st.file_uploader("")
            
            if audio_upload is not None:
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    print('tfile.name:', tfile.name)
                    tfile.write(audio_upload.read())
                    
                    audio_file = open(tfile.name, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
            
                    if st.button('Speech to text'):
                        transcript_res = speech_to_text(tfile.name)
                        st.write("Audio Transcript : ",transcript_res)

                    if st.button('Get summary'):
                        summary = summarizer(generate_speech(tfile.name))
                        c = st.container()
                        c.write("Preparing the summary for you")
                        #st.write(summary)
                        with st.spinner('Wait for it...'):
                            time.sleep(5)
                        st.success('Done!')

                        summary_text = st.text_input("Report Text", summary)
                        export_as_pdf = st.button("Export Report")

                        with st.expander("Check out the summary of your meeting"):
                            st.write(summary_text)

                        st.download_button(label="Download PDF", 
                            data=summary_text,
                            file_name="summary_download.pdf",
                            mime='application/octet-stream')

                        
                #except:
                except Exception as ex:
                    print(ex)
                    st.write("Please upload a valid file")
                    

    except:
        st.write("Choose an option")
            
