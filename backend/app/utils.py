import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS

async def on_load(manual_file_path):
    print("on_load", manual_file_path)
    pass;

async def process_query(query):
    return "TODO"