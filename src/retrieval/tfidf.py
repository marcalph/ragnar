from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from langfuse import observe

class TFIDFRetriever():                                                                                
    vectorizer: TfidfVectorizer = TfidfVectorizer()                                                           
    index: list = None                                                                                        
    data: list = None                                                                                         
                                                                                                              
    def index_data(self, data):                                                                               
        """                                                                                                   
        Indexes the provided data using TF-IDF.                                                               
                                                                                                              
        Args:                                                                                                 
            data (list): A list of documents to be indexed. Each document should be a dictionary              
                         containing a key 'cleaned_content' with the text to be indexed.                      
        """                                                                                                   
        self.data = data                                                                                      
        docs = [doc["cleaned_content"] for doc in data]                                                       
        self.index = self.vectorizer.fit_transform(docs)                                                      

    @observe()                                                                                                              
    def search(self, query, k=5):                                                                             
        """                                                                                                   
        Searches the indexed data for the given query using cosine similarity.                                
                                                                                                              
        Args:                                                                                                 
            query (str): The search query.                                                                    
            k (int): The number of top results to return. Default is 5.                                       
                                                                                                              
        Returns:                                                                                              
            list: A list of dictionaries containing the source, text, and score of the top-k results.         
        """                                                                                                   
        query_vec = self.vectorizer.transform([query])                                                        
        cosine_distances = cdist(                                                                             
            query_vec.todense(), self.index.todense(), metric="cosine"                                        
        )[0]                                                                                                  
        top_k_indices = cosine_distances.argsort()[:k]                                                        
        output = []                                                                                           
        for idx in top_k_indices:                                                                             
            output.append(                                                                                    
                {                                                                                             
                    "source": self.data[idx]["metadata"]["source"],                                           
                    "text": self.data[idx]["cleaned_content"],                                                
                    "score": 1 - cosine_distances[idx],                                                       
                }                                                                                             
            )                                                                                                 
        return output                                                                                         
                                                                                                              
    @observe()
    def predict(self, query: str, k: int):                                                                    
        """                                                                                                   
        Predicts the top-k results for the given query.                                                       
                                                                                                              
        Args:                                                                                                 
            query (str): The search query.                                                                    
            k (int): The number of top results to return.                                                     
                                                                                                              
        Returns:                                                                                              
            list: A list of dictionaries containing the source, text, and score of the top-k results.         
        """                                                                                                   
        return self.search(query, k)  