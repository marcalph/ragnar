from utils.tokenize import get_special_tokens_set
from loguru import logger

def main(chunked_data):
    cleaned_data = []
    for doc in chunked_data:
        cleaned_doc = doc.copy()
        cleaned_doc["cleaned_content"] = make_text_tokenization_safe(doc["content"])
        cleaned_doc["metadata"]["cleaned_tokens"] = len(
            cleaned_doc["cleaned_content"].split()
        )
        cleaned_data.append(cleaned_doc)
    logger.info(f"#of cleaned chunks: {len(cleaned_data)}")
    return cleaned_data



def make_text_tokenization_safe(                                                                              
    content: str, special_tokens_set: set = get_special_tokens_set()                                          
    ) -> str:                                                                                                     
    """                                                                                                       
    Makes the text safe for tokenization by removing special tokens.                                          
                                                                                                            
    Args:                                                                                                     
        content: A string containing the text to be processed.                                                
        special_tokens_set: A set of special tokens to be removed from the text.                              
                                                                                                                
    Returns:                                                                                                  
        A string with the special tokens removed.                                                             
    """                                                                                                       
                                                                                                                 
    def remove_special_tokens(text: str) -> str:                                                              
        """                                                                                                   
        Removes special tokens from the given text.                                                           
                                                                                                                
        Args:                                                                                                 
            text: A string representing the text.                                                             
                                                                                                                
        Returns:                                                                                              
            The text with special tokens removed.                                                             
        """                                                                                                   
        for token in special_tokens_set:                                                                      
            text = text.replace(token, "")                                                                    
        return text                                                                                           
                                                                                                                
    cleaned_content = remove_special_tokens(content)                                                          
    return cleaned_content