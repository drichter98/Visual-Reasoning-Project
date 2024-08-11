import json
import time
import requests
import base64
import logging
import re

import firebase_admin
from firebase_admin import credentials, db, storage

from Constants import JSON_FILE_PATH
from generate_content import generate_subquestion_initial, final_reasoning, generate_conclusion, answer_subquestion, generate_subquestion, get_vlm_final_reasoning, generate_baseline
from LLama_model import send_request_to_Llama
from Llava_model import send_request_to_Llava
from prepare_data import prepare_data_for_llms_batched
from prompts import P1_VLM_SYSTEM

# Prepare dataset from JSON file path
dataset = prepare_data_for_llms_batched(JSON_FILE_PATH)
no_questions = 5  # number of allowed questions per reasoner
count_correct_answer = 0
exception_counter = 0
total_time_elapsed = 0

count_vlm_correct_answer = 0

# Configure logging to output to a file
logging.basicConfig(filename='output', level=logging.INFO, format='%(message)s')

# Initialize Firebase with credentials
cred = credentials.Certificate(r'') #set path of credentials
firebase_admin.initialize_app(cred, {
    'databaseURL': '', #set database url
    'storageBucket': '' #set storage bucket url
})
bucket = storage.bucket()

# Initialize the interval for uploading logs
upload_interval = 1000

# Main function to run inference
def main():
    global count_correct_answer, upload_interval, total_time_elapsed, exception_counter, count_vlm_correct_answer

    logging.info("Starting inference...")
    for idx, datapoint in enumerate(dataset):
        logging.info(f"############################# START OF DATAPOINT {idx+1} #############################")

        try:
            start_time = time.time()  # Start the timer
            count_correct_answer = do_visual_reasoning(datapoint, count_correct_answer, idx)
            elapsed_time = time.time() - start_time  # End the timer

            total_time_elapsed += elapsed_time
            
            logging.info(f"Total correct answers: {count_correct_answer}")
            logging.info(f"Total wrong answers: {idx+1 - count_correct_answer - exception_counter}")
            logging.info(f"Total errors: {exception_counter}")
            logging.info(f"Current Accuracy: {count_correct_answer / (idx+1):.4f}")

            logging.info(f"\nTime elapsed for do_visual_reasoning: {elapsed_time:.2f} seconds. In total: {total_time_elapsed:.2f} seconds. On average: {(total_time_elapsed / (idx+1)):.2f} seconds.")

            # Getting baseline (VLM) answer
            logging.info("\n")
            count_vlm_correct_answer = baseline(idx, datapoint, count_vlm_correct_answer)
            logging.info(f"Total VLM correct answers: {count_vlm_correct_answer}")
            logging.info(f"Current VLM Accuracy: {count_vlm_correct_answer / (idx+1):.4f}")
            
        except Exception as e:
            exception_counter += 1
            # Catching the exception and logging the error message
            logging.error(f"Exception occurred at datapoint {idx}: {e}", exc_info=True)
            # Optionally, log more details about the datapoint that caused the error
            logging.debug(f"Datapoint causing error: {datapoint}")

        logging.info(f"############################# END OF DATAPOINT {idx+1} #############################\n\n")

def do_visual_reasoning(datapoint: list, count_correct_answer: int, idx: int):
    global upload_interval
    conclusions = []  # contains conclusions of Reasoners (Yes/No, Reasoning)

    # Generate VLM caption
    caption = send_request_to_Llava(P1_VLM_SYSTEM, datapoint[0]["photo_filename"])
    logging.info(f"VLM Caption: {caption}")

    # Ask question to vlm direcly 
    question_for_vlm = get_vlm_final_reasoning(datapoint[0]["question"],datapoint[0]["answer_choice"],datapoint[1]["answer_choice"],datapoint[2]["answer_choice"],datapoint[3]["answer_choice"])
    vlm_answer = send_request_to_Llava(question_for_vlm, datapoint[0]["photo_filename"])
    logging.info(f"VLM Answer: {vlm_answer}")

    # Do reasoning of the 4 Reasoners (LLMs)
    for answer_choice in datapoint:
        sub_questions = []
        subq_answers = []
        logging.info(f"\nAnswer choice: {answer_choice['answer_choice']}")

        # Run conversation between Reasoner and VLM
        for i in range(no_questions):
            if i == 0:
                # generate first subquestion
                P2_I1_SYSTEM, P2_I1_INPUT = generate_subquestion_initial(answer_choice['question'], caption,
                                                                                          answer_choice['answer_choice'])
                sub_question_1 = send_request_to_Llama(
                    [{"role": "system", "content": P2_I1_SYSTEM}, {"role": "user", "content": P2_I1_INPUT}])
                sub_questions.append(sub_question_1) # save subquestions as conversation history
                logging.info(f"LLM 1: {sub_question_1}")

                # get answer to subquestion
                answer_1 = send_request_to_Llava(prompt=answer_subquestion(sub_question_1.replace("Sub-question: ", "")),
                                               image=answer_choice["photo_filename"])
                logging.info(f"VLM 1: {answer_1}")
                subq_answers.append(answer_1) # save answers as conversation history
            else:
                # generate subsequent subquestion
                P2_FI_SYSTEM, P2_FI_INPUT = generate_subquestion(answer_choice['question'], caption,
                                                                                               answer_choice['answer_choice'],
                                                                                               sub_questions,
                                                                                               subq_answers)
               
                sub_question = send_request_to_Llama(
                    [{"role": "system", "content": P2_FI_SYSTEM}, {"role": "user", "content": P2_FI_INPUT}])
                sub_questions.append(sub_question) # save subquestions as conversation history
                logging.info(f"LLM {i+1}: {sub_question}")

                # get answer to subquestion
                answer = send_request_to_Llava(prompt=answer_subquestion(sub_question.replace("Additional sub-question: ", "")),
                                               image=answer_choice["photo_filename"])
                subq_answers.append(answer) # save answers as conversation history
                logging.info(f"VLM {i+1}: {answer} ")

        # Generate prompt for conclusion finding of reasoners
        P4_SYSTEM, P4_INPUT = generate_conclusion(answer_choice['question'], caption, answer_choice['answer_choice'],
                                                                    sub_questions, subq_answers)
        conc = send_request_to_Llama(
            [{"role": "system", "content": P4_SYSTEM}, {"role": "user", "content": P4_INPUT}])
        logging.info(f"Conclusion: {conc} \n")
        conclusions.append(answer_choice)
        conclusions.append(conc)

    # Generate prompt for final reasoning
    P5_SYSTEM, P5_INPUT = final_reasoning(datapoint[0]['question'], caption, conclusions[0], conclusions[1],
                                                     conclusions[2], conclusions[3], conclusions[4], conclusions[5],
                                                     conclusions[6], conclusions[7],vlm_answer)
    reasoning = send_request_to_Llama(
        [{"role": "system", "content": P5_SYSTEM}, {"role": "user", "content": P5_INPUT}])
    logging.info(f"\nOverall analysis: {reasoning}")

    answer_index = extract_final_answer_index(reasoning) # final answer index of answer choice
    
    prediction = datapoint[answer_index-1]["answer_choice"] # based on index extract choice
    logging.info(f"\nPredicted answer: {prediction}")

    label = int(datapoint[0]["correct_choice_idx"]) # get true label
    true_answer = datapoint[label]["answer_choice"]
    logging.info(f"True answer: {true_answer}")

    if answer_index == label + 1:
        count_correct_answer += 1
        logging.info("Final: Correct answer\n\n")
    else:
        logging.info("Final: Wrong answer\n\n")
    

    # Upload output log file to Firebase Storage based on dynamic interval
    if idx == upload_interval - 100:
        # Send results to Firebase Realtime Database
        send_results_to_firebase_realtime(idx, count_correct_answer)
        upload_output_log_to_storage(idx)
        # Increment the upload interval
        upload_interval += 200

    return count_correct_answer

def send_results_to_firebase_realtime(idx, count_correct_answer):
    """
    Sends the results to Firebase Realtime Database.

    Parameters:
    idx (int): Index of the current datapoint.
    count_correct_answer (int): Number of correct answers so far.
    """
    ref = db.reference(f'results/{idx}')
    ref.set({
        'idx': idx,
        'count_correct_answer': count_correct_answer
    })
    logging.info(f"Results sent to Firebase Realtime Database: idx={idx}, count_correct_answer={count_correct_answer}")

def upload_output_log_to_storage(idx):
    """
    Uploads the output log file to Firebase Storage.

    Parameters:
    idx (int): Index of the current datapoint.
    """
    blob = bucket.blob(f'output_logs/output_{idx}.txt')
    blob.upload_from_filename('output')
    logging.info(f"Output log uploaded to Firebase Storage: output_{idx}.txt")


def extract_final_answer_index(text):
    """
    Extracts the final answer index from the text.

    Parameters:
    text (str): The text containing the final answer index.

    Returns:
    int: The extracted answer index.
    """
    pattern = r"Final Answer: (\d+)"

    match = re.search(pattern, text)
    if match:
        number = match.group(1)  # Extract the first capturing group
        return int(number)
    return -1

def baseline(idx, datapoint, count_vlm_correct_answer):
    """
    Computes the baseline (VLM) answer and compares it to the correct answer.

    Parameters:
    idx (int): Index of the current datapoint.
    datapoint (list): List of datapoint containing the question and answer choices.
    count_vlm_correct_answer (int): Number of correct VLM answers so far.

    Returns:
    int: Updated number of correct VLM answers.
    """
    question = datapoint[0]['question']
    a1 = datapoint[0]['answer_choice']
    a2 = datapoint[1]['answer_choice']
    a3 = datapoint[2]['answer_choice']
    a4 = datapoint[3]['answer_choice']
    result = send_request_to_Llava(generate_baseline(question,a1,a2,a3,a4), datapoint[0]["photo_filename"])
    answer = datapoint[int(result)-1]['answer_choice']
    logging.info(f"VLM Answer: {answer}")
    correct_label = int(datapoint[0]["correct_choice_idx"]) + 1
    
    if int(result) == correct_label:
        count_vlm_correct_answer += 1
    
    return count_vlm_correct_answer

main()



