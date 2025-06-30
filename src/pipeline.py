from generation.response import SimpleResponseGenerator
from retrieval.tfidf import TFIDFRetriever

from dataclasses import dataclass

@dataclass
class SimpleRAGPipeline():                                                                         
    """                                                                                                       
    A simple RAG (Retrieval-Augmented Generation) pipeline.                                                   
                                                                                                              
    Attributes:                                                                                               
        retriever (weave.Model): The model used for retrieving relevant documents.                            
        response_generator (weave.Model): The model used for generating responses.                            
        top_k (int): The number of top documents to retrieve.                                                 
    """                                                                                                                                                                                                              
    retriever: TFIDFRetriever|None = None                                                                             
    response_generator: SimpleResponseGenerator| None = None                                                                    
    top_k: int = 5                                                                                            
                                                                                                              
    def predict(self, query: str):                                                                            
        """                                                                                                   
        Predicts a response based on the input query.                                                         
                                                                                                              
        Args:                                                                                                 
            query (str): The input query string.                                                              
                                                                                                              
        Returns:                                                                                              
            The generated response based on the retrieved context.                                            
        """                                                                                                   
        context = self.retriever.predict(query, self.top_k)                                                   
        return self.response_generator.predict(query, context)

