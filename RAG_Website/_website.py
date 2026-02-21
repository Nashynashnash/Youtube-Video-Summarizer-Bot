import streamlit as st
import base64
from RAG.transcript import get_transcript
import re
from youtube_transcript_api import YouTubeTranscriptApi

with open("RAG_Website/bgm.png", "rb") as f:
    png_data = f.read()
b64 = base64.b64encode(png_data).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)   
import streamlit as st

st.markdown('<h1 style="color:black;">MyGPT</h1>', unsafe_allow_html=True)   

st.markdown(
    '<p style="color:black;">The best Youtube Video summarizer with the help of AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="color:black;">Your Youtube video link here',
    unsafe_allow_html=True
)
input = st.text_input('')

if st.button('Enter') and input:
    st.session_state['input'] = get_transcript(input)
    st.switch_page("pages/Chat Bot.py") 



