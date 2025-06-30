from constants import setup, DATA_PATH
from loguru import logger
from preprocessing.parse import main as parse
from preprocessing.chunk import main as chunk
from preprocessing.clean import main as clean
from retrieval.tfidf import TFIDFRetriever



def list_docs(data_path):
    """list relevant md files from a directory"""
    docs_files = sorted(data_path.rglob("*.md"))
    logger.info(f"Number of files: {len(docs_files)}\n")
    logger.debug("First 3 files:\n{files}".format(files="\n".join(map(str, docs_files[:3]))))
    logger.debug(docs_files[0].read_text())
    return docs_files




if __name__== "__main__":
    docs_dir = DATA_PATH
    docs_files = list_docs(docs_dir)
    data = parse(docs_files)
    total_tokens = sum(map(lambda x: x["metadata"]["raw_tokens"], data))
    logger.info(f"#of tokens in dataset: {total_tokens}")
    chunked_data = chunk(data)
    cleaned_data = clean(chunked_data)
    retriever = TFIDFRetriever()
    retriever.index_data(cleaned_data)
    query = "How do I use W&B to log metrics in my training script?"
    search_results = retriever.search(query)
