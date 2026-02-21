import streamlit as st
import base64
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# ---------- EXTRACT VIDEO ID ----------
def get_match(url):
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else ""


# ---------- TRANSCRIPT FUNCTION ----------
@st.cache_data(ttl=3600)
def get_transcript(url):
    yt_api = YouTubeTranscriptApi()
    if not url:
        return "", "Empty URL"

    video_code = get_match(url)
    if not video_code:
        return "", "Invalid URL"

    try:
        # Works in ALL versions
        transcript = yt_api.fetch(video_code, languages=['en'])

        main_script = " ".join([item['text'] for item in transcript])

        return main_script, "Success"

    except Exception as e:
        return "", str(e)


# ---------- BACKGROUND ----------
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

# ---------- UI ----------
st.markdown('<h1 style="color:black;">MyGPT</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:black;">The best Youtube Video summarizer with AI</p>', unsafe_allow_html=True)
st.markdown('<p style="color:black;">Your Youtube video link here</p>', unsafe_allow_html=True)

url = st.text_input('')

if st.button('Enter') and url:

    transcript, status = get_transcript(url)

    if status != "Success":
        st.error(f"Reason: {status}")
    else:
        st.session_state["transcript"] = transcript
        st.session_state["video_id"] = get_match(url)
        st.switch_page("pages/Chat Bot.py")
