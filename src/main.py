from config import setup, DATA_PATH, MODEL
from loguru import logger
from preprocessing.parse import main as parse
from preprocessing.chunk import main as chunk
from preprocessing.clean import main as clean
from pipeline import SimpleRAGPipeline
from retrieval.tfidf import TFIDFRetriever
from generation.response import SimpleResponseGenerator
from opentelemetry.instrumentation.cohere import CohereInstrumentor



from generation.prompts import INITIAL_PROMPT

def list_docs(data_path):
    """list relevant md files from a directory"""
    docs_files = sorted(data_path.rglob("*.md"))
    logger.info(f"Number of files: {len(docs_files)}\n")
    logger.debug("First 3 files:\n{files}".format(files="\n".join(map(str, docs_files[:3]))))
    logger.debug(docs_files[0].read_text())
    return docs_files



if __name__== "__main__":
    CohereInstrumentor().instrument()
    langfuse = setup()
    docs_files = list_docs(DATA_PATH)
    data = parse(docs_files)
    total_tokens = sum(map(lambda x: x["metadata"]["raw_tokens"], data))
    logger.info(f"#of tokens in dataset: {total_tokens}")
    chunked_data = chunk(data)
    cleaned_data = clean(chunked_data)
    retriever = TFIDFRetriever()
    retriever.index_data(cleaned_data)
    response_generator = SimpleResponseGenerator(model=MODEL, prompt=INITIAL_PROMPT)
    rag = SimpleRAGPipeline(
    retriever=retriever, response_generator=response_generator, top_k=5
    )
    query = "How do I use W&B to log metrics in my training script?"
    response = rag.predict(query)
    logger.warning(f"input query {query}")
    logger.warning(f"generated answer {response}")
    langfuse.shutdown()