import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_DOC_PATH = os.getenv("EMBEDDING_DOC_PATH")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")