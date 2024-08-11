import json
import requests
from Constants import LLAMA_INFERENCE

def send_request_to_Llama(messages: list[dict]) -> str:
    """
    Sends a message to the LLaMA model and returns the response text.

    Parameters:
    messages (list[dict]): A list of submessage dictionaries (always role & content) to send to the model.

    Returns:
    str: The response text from the LLaMA model.
    """

    # Construct LLaMA message
    llama_message = {
        "messages": messages,
        "temperature": 0.7, # 0.3 was before
        "top_p": 0.4,
        "max_new_tokens": 256
    }
    
    # Send to LLaMA
    llama_response = requests.post(LLAMA_INFERENCE, json=llama_message)
    llama_response_text = llama_response.text  # Get the response as text

    # Handle LLaMA response text
    try:
        llama_response_json = json.loads(llama_response_text)
    except json.JSONDecodeError:
        llama_reply = llama_response_text

    return llama_response_text