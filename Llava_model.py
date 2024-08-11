import time
import requests
import base64
from Constants import LLAVA_INFERENCE, TEXT_STREAM

global response_output

def encode_image_to_base64(image_path):
    """
    Encodes an image file to a base64 string.

    Parameters:
    image_path (str): The file path of the image to be encoded.

    Returns:
    str: The base64 encoded string of the image.
    """
    with open(image_path, 'rb') as image_file:
        return str(base64.b64encode(image_file.read()).decode('utf-8'))

def send_request_to_Llava(prompt: str, image: str) -> str:
    """
    Sends a prompt and an image to the LLAVA inference model and returns the response.

    Parameters:
    prompt (str): The text prompt to send to the model.
    image (str): The file path of the image to send to the model.

    Returns:
    str: The response text from the LLAVA model.
    """
    # Prepare the payload with the prompt and encoded image
    payload = {
        'model_path': 'liuhaotian/llava-v1.6-mistral-7b',
        'image_base64': encode_image_to_base64(image),
        'prompt': prompt,
        'temperature': 0.0, # 0.2 was before
        'max_new_tokens': 2048, #512 before
        'stream': TEXT_STREAM
    }

    # Send the payload to the LLAVA inference endpoint
    r = requests.post(
        f'{LLAVA_INFERENCE}/inference',
        json=payload,
        stream=TEXT_STREAM,
    )

    # Handle the streaming response if TEXT_STREAM is True
    if TEXT_STREAM:
        if r.encoding is None:
            r.encoding = 'utf-8'
    else:
        resp_json = r.json()

    # Store the response in the global variable
    response_output = resp_json["response"]

    return response_output