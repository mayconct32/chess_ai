import os

from dotenv import load_dotenv


load_dotenv()

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
API_KEY = os.getenv("API_KEY")
GEMINI_VERSION = "gemini-3.1-flash-lite"
DATABASE_DIR = "database"
