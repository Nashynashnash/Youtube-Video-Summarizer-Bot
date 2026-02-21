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

load_dotenv()

st.title('MyGPT')

embed_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api
)

vector_store = Chroma(
    collection_name='samples',
    embedding_function=embed_model,
    persist_directory="RAG_Website/vector_store"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

# Reset DB when new video comes
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

if st.session_state.last_transcript != transcript:
    vector_store.delete_collection()
    st.session_state.vector_ready = False
    st.session_state.last_transcript = transcript

# Embed only once
if "vector_ready" not in st.session_state or not st.session_state.vector_ready:
    split_docs = splitter.split_text(transcript)
    docs = [Document(page_content=t) for t in split_docs]
    vector_store.add_documents(docs)
    st.session_state.vector_ready = True


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


# Create columns for dropdowns and chat input
col1, col2, col3 = st.columns([2, 2, 4])

with col1:
    response_ = st.selectbox('Response format:', ['Bullet Points', 'Paragraph format'])
with col2:
    length_ = st.selectbox('Length:', ['Concise one line', 'Few Lines', 'Detailed Paragraph', 'Detailed Report'])
with col3:
    query = st.chat_input("Ask a question")   
    
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process new user input

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


  
