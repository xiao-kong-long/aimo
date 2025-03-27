from peft import get_peft_model, LoraConfig, TaskType  # 引入 PEFT 库
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Features, Value, Dataset, DatasetDict
from sft import read_jsonl_file
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import logging
from sft import download_and_prepare_data

MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B"  # 可以换成任意支持的模型
# MODEL_NAME = "gpt2"

SYSTEM_PROMPT = """
You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. You should output in a format similar to <think>...</think><answer>...</answer>, where <think> contains the reasoning process and <answer> contains the final answer. Return final answer within \\boxed{}, after taking modulo 1000. 
"""

train_data_list = read_jsonl_file("math_data/V1_filtered/train_data_filtered.jsonl")[:1000]
train_data = Dataset.from_dict({key: [d[key] for d in train_data_list] for key in train_data_list[0].keys()})
val_data_list = read_jsonl_file("math_data/V1_filtered/test_small_data_filtered.jsonl")
val_data = Dataset.from_dict({key: [d[key] for d in val_data_list] for key in val_data_list[0].keys()})

dataset = DatasetDict({"train": train_data, "validation": val_data})

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dist.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.getenv("LOCAL_RANK", 0))  # 从环境变量中获取 local_rank
torch.cuda.set_device(local_rank)  # 设置当前进程使用的 GPU

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(local_rank)

# 配置 LoRA 参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型为因果语言模型
    r=4,  # LoRA 的秩
    lora_alpha=32,  # LoRA 的缩放因子
    lora_dropout=0.1,  # LoRA 的 Dropout 概率
    target_modules=["q_proj", "v_proj"]  # 指定需要应用 LoRA 的模块
)

# 将模型转换为 LoRA 模型
model = get_peft_model(base_model, lora_config)
model = model.to(local_rank)

# 打印可训练参数数量
model.print_trainable_parameters()

# 数据预处理和训练逻辑保持不变
tokenizer.pad_token = tokenizer.eos_token

# # 数据预处理函数
# def tokenize_function(examples):
#     tokenized = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=128)
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     print(len(tokenized["labels"]))
#     print(len(tokenized["labels"][0]))
#     return tokenized

# 数据预处理函数
def tokenize_function(examples, max_length=512):
    # 将 prompt 和 answer 拼接为输入
    prompts = [ SYSTEM_PROMPT + "\n" + example for example in examples["prompt"] ]
    answers = []
    for answer in examples["answer"]:
        try:
            answers.append(int(answer) % 1000)
        except Exception:
            answers.append(0)
    
    answer_texts = [f"<think>{think}</think>\n" \
                  f"<answer>{answer}. So the mod 1000 answer is \\boxed{{{str(answer_mod)}}}</answer>" for think, answer, answer_mod in zip(examples["think"], examples['answer'], answers)]

    input_text = [prompt + " " + answer for prompt, answer in zip(prompts, answer_texts)]
    tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_length)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

    assert any([len(tokenized["labels"][i]) == max_length for i in range(len(tokenized["labels"]))])

    # 计算每个 input_text 的 token 数量
    # input_token_lengths = [len(tokenizer(text, truncation=True)["input_ids"]) for text in input_text]
    # print("max input token length: ", max(input_token_lengths))
    # print("average input token length: ", sum(input_token_lengths) / len(input_token_lengths))
    # print("the number over max length: ", len([length for length in input_token_lengths if length > max_length]))

    # 计算每个 prompt 的 token 数量
    prompt_token_lengths = [len(tokenizer(prompt, truncation=True)["input_ids"]) for prompt in prompts]

    # 将非 answer 部分的 token 设置为 -100，避免计算损失
    index = 0
    for prompt_token_length, answer in zip(prompt_token_lengths, answers):
        if prompt_token_length > len(tokenized['labels'][index]):
            logging.warning("Prompt too long, the length is %d", prompt_token_length)
            continue
        tokenized["labels"][index][:prompt_token_length] = [-100]*prompt_token_length
        index += 1

    assert any([len(tokenized["labels"][i]) == len(tokenized['input_ids'][i]) for i in range(len(tokenized["labels"]))])

    return tokenized

# 预处理数据集
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# train_dataset = download_and_prepare_data("math_data/V1_filtered/train_data_filtered.jsonl", tokenizer=tokenizer, batch_size=4).dataset
# val_dataset = download_and_prepare_data("math_data/V1_filtered/test_small_data_filtered.jsonl", tokenizer=tokenizer, batch_size=4).dataset
# tokenized_datasets = {"train": train_dataset, "validation": val_dataset}

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    deepspeed="aimo/ds_config.json",  # 指定 DeepSpeed 配置文件
    fp16=True,  # 启用 FP16 精度
    report_to="none",  # 不使用 WandB 等外部报告
    remove_unused_columns=False
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# 启动训练
trainer.train()

# Save the trained model and tokenizer
final_checkpoint_path = "finetuned_model/deepseek-lora/final"
os.makedirs(final_checkpoint_path, exist_ok=True)
model.save_pretrained(final_checkpoint_path)
tokenizer.save_pretrained(final_checkpoint_path)