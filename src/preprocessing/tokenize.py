from config import TOKENIZERS
import httpx

def get_special_tokens_set(tokenizer_url=TOKENIZERS["command-r"]):
    """
    Fetches the special tokens set from the given tokenizer URL.

    Args:
        tokenizer_url (str): The URL to fetch the tokenizer from.

    Returns:
        set: A set of special tokens.
    """
    # https://docs.cohere.com/docs/tokens-and-tokenizers
    response = httpx.get(tokenizer_url)
    return set([tok["content"] for tok in response.json()["added_tokens"]])
