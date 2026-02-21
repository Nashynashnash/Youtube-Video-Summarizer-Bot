import streamlit as st
from langchain_anthropic import ChatAnthropic
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

#load_dotenv()

#video_url = "https://www.youtube.com/watch?v=9-GRzu6zbS0"
#query = 'what is the main point of this video'

st.title('MyGPT')

transcript = st.session_state['input']
query = st.chat_input('Write what you want to understand about the video')
api = 'AIzaSyAUEhcCt6K9ttkS5bxguFzVL7JMhoJxccU'

embed_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api)
my_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=  api
)

vector_store = Chroma(
        collection_name = 'samples',
        embedding_function = embed_model,
        persist_directory = "RAG_Website/vector store"
)
splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators=''
    )
split_docs = splitter.split_text(transcript)
vector_store.add_texts(texts=split_docs)



def get_output(docs, query, response_, length_):

    #my_llm = ChatHuggingFace(llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-1B-Instruct",task = "text-generation"))
    #embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #my_llm = ChatAnthropic(model_name='claude-3-sonnet-20240229')
    base_retriever = vector_store.as_retriever(search_type = 'mmr',search_kwargs = {'k':5, 'lambda_mult': 0.8})

    # mqr_retriever = MultiQueryRetriever.from_llm(
    #     retriever = base_retriever , 
    #     llm = my_llm
    # )

    # compressor = LLMChainExtractor.from_llm(llm=my_llm)

    # final_retriever = ContextualCompressionRetriever(
    #     base_retriever = mqr_retriever,
    #     base_compressor = compressor
    # )

    template = """
You are a YouTube video summarizer AI. Generate a concise and accurate summary using only the information in the transcript below.

Instructions:
- Focus on main topic, key points, and important details.
- Ignore filler words, repetitions, and irrelevant content.
- If the question cannot be answered from the transcript, respond with: "I don't have enough information in the provided context to answer this."

Question: {question}

Transcript: {text}

Response Type: Give it in detailed {response_} 

Length of Response: Answer it in the following {length_}
"""

    prompt_template = PromptTemplate(
        template=template,
        input_variables=['question', 'text', 'response_', 'length_'])   

    context = "\n".join([doc.page_content for doc in base_retriever.invoke(query)])

    final_template = prompt_template.invoke({'question':query, 'text':context, 'response_': response_,'length_':length_})

    result = my_llm.invoke(final_template)

    print(prompt_template)

    return result.content

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process new user input

response_ = st.selectbox('choose reponse format:',['Bullet Points', 'Paragraph format'])
length_ = st.selectbox('Response Length',['Concise one line answer', 'Few Lines', 'Detailed Paragraph', 'Detailed Report'])
if query:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    


    # Generate and display assistant response
    if query == 'hi':
        output = 'Hello there. What are we studying today?'
    else:
        output = get_output(transcript, query, response_=response_, length_=length_)
    
    st.session_state.messages.append({"role": "assistant", "content": output})

    
    with st.chat_message("assistant"):
        st.write(output) 


  