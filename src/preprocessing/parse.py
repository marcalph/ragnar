from config import DATA_PATH
from loguru import logger

def main(documents):
    """"Parse documents list into hashmap"""
    data = []
    for file in documents:
        content = file.read_text()
        data.append(
            {
                "content": content,
                "metadata": {
                    "source": str(file.relative_to(DATA_PATH)),
                    "raw_tokens": len(content.split()),
                },
            }
        )
    logger.info(f"#of docs in dataset: {len(data)}")
    return data