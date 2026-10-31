import os 
from dotenv import load_dotenv
from langchain import OpenAI
from langchain_openai import OpenaiEmbeddings

load_dotenv()


embeddings = OpenaiEmbeddings(model="text-embedding-3-large")
