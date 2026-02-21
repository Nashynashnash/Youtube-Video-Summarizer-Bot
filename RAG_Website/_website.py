import streamlit as st
import base64
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

def get_match(url):
     if not url:
        return ""

     match = re.search(r"(?:v=|youtu\.be/)(.{11})", url)
     if not match:
        return ""
     return match.group(1)

def get_transcript(url):
    if not url:
        return ""

    match = re.search(r"(?:v=|youtu\.be/)(.{11})", url)
    if not match:
        return ""

    video_code = match.group(1)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_code, languages=['en'])
    except:
        return "Transcript not available"

    main_script = " ".join([script["text"] for script in transcript])
    return main_script


# background image
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

# UI
st.markdown('<h1 style="color:black;">MyGPT</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:black;">The best Youtube Video summarizer with AI</p>', unsafe_allow_html=True)
st.markdown('<p style="color:black;">Your Youtube video link here</p>', unsafe_allow_html=True)

url = st.text_input('')

if st.button('Enter') and url:
    transcript = get_transcript(url)
    st.session_state["transcript"] = transcript
    if transcript != "Transcript not available":
         st.session_state["video_id"] = get_match(url)
         st.switch_page("pages/Chat Bot.py")
