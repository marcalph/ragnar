from dotenv import load_dotenv
import wandb
from pathlib import Path

PROJECT="rag-course"
DATA_PATH=Path("rag/data/wandb_docs")

# hprms
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0
TOKENIZERS = {
    "command-r": "https://storage.googleapis.com/cohere-public/tokenizers/command-r.json",
    "command-r-plus": "https://storage.googleapis.com/cohere-public/tokenizers/command-r-plus.json",
}
MODEL="command-r"



def setup():
    load_dotenv()
    wandb.login()




