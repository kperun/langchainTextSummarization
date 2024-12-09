import os
import dotenv
from langchain_openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
#import streamlit as st

# Load the .env file
dotenv.load_dotenv('API.env')
OpenAI.api_key = os.getenv("OPEN_API_KEY")

# Load the PDF file which will be used for RAG
pdf_url = "https://www.gesetze-im-internet.de/gg/GG.pdf"

loader = PyPDFLoader(pdf_url)
pages = loader.load_and_split()

# check if document loading was correct
if pages:
    print('Number of pages: ', len(pages))
    print(pages[0].page_content)
else:
    print('Document with 0 pages loaded, exit')
    exit(0)

