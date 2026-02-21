from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import re

load_dotenv()

video_url = "https://www.youtube.com/watch?v=9-GRzu6zbS0"
#query = 'what is the main point of this video'

def get_transcript(url):
    if url == None :
        return ''
    match = re.search('v=(.{11})',url)
    video_code = match.group(1)
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_code, languages=['en'])   
    main_script = ""
    for script in transcript:
        main_script = main_script +" " + script.text

    return main_script

print(get_transcript(url=video_url))
