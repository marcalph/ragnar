import cohere
import os
import langfuse

from langfuse import observe
class SimpleResponseGenerator():
    model: str                                                                                                
    prompt: str                                                                                               
    client: cohere.ClientV2 = None                                                                            
                                                                                                              
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)                                                                     
        self.client = cohere.ClientV2(                                                                        
            api_key=os.environ["COHERE_API_KEY"],                                                             
        )                                                                                                     
                                                                                                              
    def generate_context(self, context: list[dict[str, any]]) -> list[dict[str, any]]:                        
        """                                                                                                   
        Generate a list of contexts from the provided context list.                                           
                                                                                                              
        Args:                                                                                                 
            context (list[dict[str, any]]): A list of dictionaries containing context data.                   
                                                                                                              
        Returns:                                                                                              
            list[dict[str, any]]: A list of dictionaries with 'source' and 'text' keys.                       
        """                                                                                                   
        contexts = [                                                                                          
            {"data": {"source": item["source"], "text": item["text"]}}                                        
            for item in context                                                                               
        ]                                                                                                     
        return contexts                                                                                       
                                                                                                        
    def create_messages(self, query: str):                                                                    
        """                                                                                                   
        Create a list of messages for the chat model based on the query.                                      
                                                                                                              
        Args:                                                                                                 
            query (str): The user's query.                                                                    
                                                                                                              
        Returns:                                                                                              
            list[dict[str, any]]: A list of messages formatted for the chat model.                            
        """                                                                                                   
        messages = [                                                                                          
            {"role": "system", "content": self.prompt},                                                       
            {"role": "user", "content": query},                                                               
        ]                                                                                                     
        return messages                                                                                       

    def generate_response(self, query: str, context: list[dict[str, any]]) -> str:                            
        """                                                                                                   
        Generate a response from the chat model based on the query and context.                               
                                                                                                              
        Args:                                                                                                 
            query (str): The user's query.                                                                    
            context (list[dict[str, any]]): A list of dictionaries containing context data.                   
                                                                                                              
        Returns:                                                                                              
            str: The generated response from the chat model.                                                  
        """                                                                                                   
        documents = self.generate_context(context)                                                            
        messages = self.create_messages(query)                                                                
        response = self.client.chat(                                                                          
            messages=messages,                                                                                
            model=self.model,                                                                                 
            temperature=0.1,                                                                                  
            max_tokens=2000,                                                                                  
            documents=documents,                                                                              
        )                                                                                                     
        return response.message.content[0].text                                                               

    def predict(self, query: str, context: list[dict[str, any]]):                                             
        """                                                                                                   
        Predict the response for the given query and context.                                                 
                                                                                                              
        Args:                                                                                                 
            query (str): The user's query.                                                                    
            context (list[dict[str, any]]): A list of dictionaries containing context data.                   
                                                                                                              
        Returns:                                                                                              
            str: The predicted response from the chat model.                                                  
        """                                                                                                   
        return self.generate_response(query, context)                                                         
                                                                                                              