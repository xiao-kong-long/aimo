import os
import re
import time
import json
import random
import warnings
from collections import Counter
import numpy as np, pandas as pd

import torch
import vllm
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

warnings.simplefilter('ignore')
print('PyTorch version:', torch.__version__)
print('vLLM:', vllm.__version__)

llm_model_pth = '/data/coding/upload-data/data/DeepSeek-R1-Distill-Qwen-7B'

MAX_NUM_SEQS = 32
MAX_MODEL_LEN = 8192 * 3 // 2

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""

def batch_message_filter(list_of_messages) -> tuple[list[list[dict]], list[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers

def select_answer(answers):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 'NO ANSWER'
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000

def batch_message_generate(list_of_messages) -> list[list[dict]]:
    max_tokens = MAX_MODEL_LEN
    if time.time() > cutoff_times[-1]:
        print("Speedrun")
        max_tokens = 2 * MAX_MODEL_LEN // 3

    sampling_params = SamplingParams(
        temperature=1.0,               # Randomness of the sampling
        top_p=0.90,                    # Cumulative probability of the top tokens to consider
        min_p=0.05,                    # Minimum probability for a token to be considered
        skip_special_tokens=True,      # Whether to skip special tokens in the output
        max_tokens=max_tokens,         # Maximum number of tokens to generate
        # stop=["</think>"],             # List of strings that stop the generation
        seed=777,
    )
    
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    # print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

    sort_keys_and_list_of_messages = []
    for messages, single_request_output in zip(list_of_messages, request_output):
        #print()
        # print(single_request_output.outputs[0].text)
        #print()
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )
    # print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0])
    # print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    
    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]
    return list_of_messages


def create_starter_messages(question, index):
    options = []
    for _ in range(13):
        options.append(
            [
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": question},
            ]
        )
    for _ in range(3):    
        options.append(
            [
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
                {"role": "user", "content": question},
            ],
        )
    return options[index%len(options)]

def predict_for_question(question: str) -> int:
    selected_questions_only = True
    #selected_questions_only = False
    # if selected_questions_only and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    #     #if "Triangle" not in question:
    #     #    return 210
    #     if "Triangle" not in question and "delightful" not in question and "George" not in question:
    #         return 210

    if time.time() > cutoff_time:
        return 210
    
    # print(question)

    num_seqs = MAX_NUM_SEQS
    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * MAX_NUM_SEQS // 3
    
    list_of_messages = [create_starter_messages(question, index) for index in range(num_seqs)]

    all_extracted_answers = []
    for _ in range(1):
        list_of_messages = batch_message_generate(list_of_messages)
        # df = pd.DataFrame(
        #     {
        #         "question": [question] * len(list_of_messages),
        #         "message": [messages[-1]["content"] for messages in list_of_messages],
        #     }
        # )
        # df.to_csv(f"{str(int(time.time() - start_time)).zfill(5)}.csv", index=False)
        
        list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
        all_extracted_answers.extend(extracted_answers)
    
    # print('all_extracted_answers:', all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    # print('llm answer:', answer)

    # print("\n\n")
    cutoff_times.pop()
    return answer

def predict(question):
    # id_ = id_.item(0)
    # print("------")
    # print(id_)
    # question = question.item(0)
    answer = predict_for_question(question)
    # print(question)
    # print("------\n\n")
    return answer
    # return pd.DataFrame({'id': id_, 'answer': answer})


def read_jsonl_file(file_path):
    '''
    key:
        prompt: question
        think: solution
        answer: answer
    '''
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


if __name__ == '__main__':

    seed_everything(seed=0)

    start_time = time.time()
    cutoff_time = start_time + (4 * 60 + 45) * 60
    cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 60 * 60, 50 + 1)]

    llm = LLM(
        llm_model_pth,
    #    dtype="half",                 # The data type for the model weights and activations
        max_num_seqs=MAX_NUM_SEQS,    # Maximum number of sequences per iteration. Default is 256
        max_model_len=MAX_MODEL_LEN,  # Model context length
        trust_remote_code=True,       # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
        tensor_parallel_size=2,       # The number of GPUs to use for distributed execution with tensor parallelism
        gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
        seed=2024,
    )
    print("Model loaded")
    tokenizer = llm.get_tokenizer()

    print("Ready to predict")
    data_file = 'math_data/V1_filtered/test_small_data_filtered.jsonl'

    data = read_jsonl_file(data_file)
    total = 0
    correct = 0
    for index, demo in emerate(data):
        print('index:', index, '/', len(data))
        question = demo['prompt']
        answer = demo['answer']
        try:
            answer = int(answer) % 1000
        except:
            continue

        total += 1
        if predict(question) == answer:
            correct += 1

        # print('llm answer', predict(question))
        # print('correct answer:', int(demo['answer']) % 1000)
    
    print('total:', total)
    print('correct:', correct)
    print('accuracy:', correct / total)


