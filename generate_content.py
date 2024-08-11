import prompts

# Prompt 1: To VLM: Get opinion of VLM about the right answer
# Prompt 2: To LLM: Generate first subquestion
# Prompt 3: To LLM: Generate subsequent subquestions
# Prompt 4: To VLM: Answer sub-question
# Prompt 5: LLM: Come to a conclusion
# Prompt 6: To LLM (Final Reasoner): Take all answers and reasoning and decide for the final answer
# Prompt 7: To VLM: Get answer for baseline (just VLM answer)
#################################################################


# Prompt 1: To VLM: Get opinion of VLM about the right answer
def get_vlm_final_reasoning(question, a1, a2, a3, a4):

    VLM_QUESTION = f"""You are a vision AI assistant who has strong reasoning abilities.
    You will be provided with an image.

    Your goal is to answer the multiple-choice question:
    Question: {question} 
    Answer Candidate 1: {a1}
    Answer Candidate 2: {a2}
    Answer Candidate 3: {a3}
    Answer Candidate 4: {a4}

    Here are the rules you should follow in your response:
    1. Demonstrate your reasoning and inference process within no more than 3 lines. Start with the format of "Analysis: ".
    
    Response Format:
    VLM Analysis: ...
    VLM Answer Candidate: number of candidate, name of candidate
    """
    return VLM_QUESTION

# Prompt 2: To LLM: Generate first subquestion
def generate_subquestion_initial(question, caption, answer):

    P2_I1_INPUT = f"""Main question: {question} 
    Caption: {caption} 
    Answer Candidate: {answer} 
    Please list the sub-question following the requirements I mentioned before.
    """
    return prompts.P2_I1_SYSTEM, P2_I1_INPUT

# Prompt 3: To LLM: Generate subsequent subquestions
def generate_subquestion(question, caption, answer, sub_questions, answers):
    if len(sub_questions) != len(answers):
        raise ValueError("The number of sub-questions must match the number of answers.")

    #Main part of the prompt
    P2_FI_INPUT = f"""Main question: {question}
    Caption: {caption}
    Answer candidate: {answer}
    """

    #Adding each sub-question and answer to the input
    for i, (subq, answer_subq) in enumerate(zip(sub_questions, answers), start=1):
        P2_FI_INPUT += f"\nSub-question {i} and answer from visual AI model: {subq} {answer_subq}"

    P2_FI_INPUT += "\nPlease list the sub-question following the requirement I mentioned before."

    return prompts.P2_FI_SYSTEM, P2_FI_INPUT

#Prompt 4: To VLM: Answer sub-question
def answer_subquestion(question):
    P3_INPUT = f""" You are an AI assistant with extensive visual abilities. 
    You will be provided with:

    1. An image.
    2. A question related to the image.

    Your goal is:
    To provide an answer based solely on the visual information presented in the image. Your response should be short, factual and derived directly from what you observe.

    Question: {question}
    """
    return P3_INPUT


# Prompt 5: LLM: Come to a conclusion
def generate_conclusion(question, caption, answer, sub_questions, answers):
    if len(sub_questions) != len(answers):
        raise ValueError("The number of sub-questions must match the number of answers.")
    
    P4_INPUT = f"""Main question: {question}
    Caption: {caption}
    Answer candidate: {answer}
    """
    
    for i, (subq, ans) in enumerate(zip(sub_questions, answers), start=1):
        P4_INPUT += f"\nSub-question {i}: {subq}\nAnswer {i}: {ans}"
    
    P4_INPUT += "\nPlease follow the above-mentioned instructions to list the Analysis, Answer and Confidence.\n"
    
    return prompts.P4_SYSTEM, P4_INPUT


# Prompt 6: To LLM (Final Reasoner): Take all answers and reasoning and decide for the final answer
def final_reasoning(question, caption, a1, analysis1, a2, analysis2, a3, analysis3, a4, analysis4, vlm_answer):

    P5_INPUT = f"""
    Main question: {question} 
    Caption: {caption}
    Answer of vision-language model: {vlm_answer}
    Candidate answer number 1: {a1}
    Analysis 1: {analysis1}
    Candidate answer number 2: {a2}
    Analysis 2: {analysis2}
    Candidate answer number 3: {a3}
    Analysis 3: {analysis3}
    Candidate answer number 4: {a4}
    Analysis 4: {analysis4}
    Please follow the above-mentioned instructions to provide.
    """
    return prompts.P5_SYSTEM, P5_INPUT

# Prompt 7: To VLM: Get answer for baseline (just VLM answer)
def generate_baseline(question,a1,a2,a3,a4):
    P_Baseline = f"""You are an AI assistant who has rich visual commonsense knowledge and strong reasoning abilities.
    You will be provided with an image, a question about the image and four answer candidates.
    
    Your goal is:
    To return the number of the correct answer candidate based on what you see in the image.

    Here are the rules you should follow in your response:
    1. The answer should not have any other words other than the number of the correct answer candidate.
    2. The answer should be either 1, 2, 3, or 4.
    
    Question: {question}
    Answer candidate number 1: {a1}
    Answer candidate number 2: {a2}
    Answer candidate number 3: {a3}
    Answer candidate number 4: {a4}
    """
    return P_Baseline