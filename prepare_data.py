import json
from Constants import PATH_TO_IMAGES

response_dict = {}

def prepare_data_for_llms_batched(json_file_path, batch_size=4):
    """
    Parses the JSON file, extracts relevant data, and prepares it for LLM processing in batches.
    This function for pre-processing dataset. It takes JSON file from /workspace/json/aokvqa_v1p0_val.json and process it by taking: question, choices and image_id, question_id out of it.

    Args:
        json_file_path (str): Path to the JSON file containing the image/question data.
        batch_size (int, optional): The size of each batch. Defaults to 4.

    Yields:
        list: A list of dictionaries, each containing information for a single LLM instance within a batch.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        all_batches = []

        for entry in data:
            question = entry["question"]
            choices = entry["choices"]
            image_id = entry["image_id"]
            question_id = entry["question_id"]
            correct_choice_idx = entry["correct_choice_idx"]

            photo_filename = get_photo_filename(image_id)

            current_batch = []

            for idx, answer_choice in enumerate(choices):
                llm_input = {
                    "photo_filename": photo_filename,
                    "question_id": question_id,
                    "question": question,
                    "answer_choice": answer_choice,
                    "choice_index": idx,
                    "correct_choice_idx" : correct_choice_idx
                }
                current_batch.append(llm_input)

            all_batches.append(current_batch)
        return all_batches


def get_photo_filename(image_id):
    """
    Finds the image from path and then sent this image to the LLaVA model.

    Args:
        image_id (int): The image ID.

    """

    image_id = f"{image_id:012d}"
    path_to_image = PATH_TO_IMAGES + f"/{image_id}.jpg"
    return path_to_image

def create_question_answer_pair(dataset):

    """
    It second stage of pre-processing which takes questions and answer choices and then appends them to the input which then will be provided to the LLM.

    Args:
        image_id (int): The image ID.

    """
    try:
        batch = next(dataset)
        responses = []
        for input_data in batch:
            responses.append(f"Main question: {input_data['question']}, Answer candidate: {input_data['answer_choice']}")
        for idx, response in enumerate(responses):
            response_key = f"response_{idx + 1}"
            response_dict[response_key] = response
        return response_dict
    except StopIteration:
        print("All questions have been processed")
        return None