from config import CHUNK_SIZE, CHUNK_OVERLAP
from loguru import logger
from typing import Any
import sys
# set log level
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Only log INFO and above


def main(raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunked_data = []
    for doc in raw_data:
        chunks = split_into_chunks(doc["content"])
        for chunk in chunks:
            chunked_data.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source": doc["metadata"]["source"],
                        "raw_tokens": len(chunk.split()),
                    },
                }
            )
    logger.info(f"#of chunks in dataset: {len(chunked_data)}")
    return chunked_data
    


def split_into_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Function to split the text into chunks of a maximum number of tokens
    ensure that the chunks are of size CHUNK_SIZE and overlap by chunk_overlap tokens
    use the `tokenizer.encode` method to tokenize the text
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        start = end - chunk_overlap
    return chunks