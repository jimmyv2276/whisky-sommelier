import os
from pathlib import Path

# 1. Base Directory Setup
# Path(__file__).resolve().parent ensures paths remain correct no matter where you run the project from
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Ensure the data folder always exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. File Path Definitions
CSV_FILE_PATH = DATA_DIR / "cleaned_whiskies.csv"
EMBEDDINGS_FILE_PATH = DATA_DIR / "embeddings.npy"

# 3. Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 4. API Key & LLM Setup
# In local environments, we read from system environment variables or a .env file.
# Locally, you would install `python-dotenv` and add `from dotenv import load_dotenv; load_dotenv()`
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = "gpt-4o-mini"  # You can switch between 'gpt-4o-mini', 'gpt-4o', etc.
