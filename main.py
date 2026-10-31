import os 
from dotenv import load_dotenv
from langchain import OpenAI
from langchain_openai import OpenaiEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenaiEmbeddings(model="text-embedding-3-large")


vector_store = Chroma(
    collection_name="resume_collection", 
    embedding_function=embeddings,
    persist_directory="./backend/data/chroma"
)


