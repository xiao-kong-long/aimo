from peft import get_peft_model, LoraConfig, TaskType  # 引入 PEFT 库
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Features, Value, Dataset, DatasetDict
from sft import read_jsonl_file
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B"  # 可以换成任意支持的模型
# MODEL_NAME = "gpt2"

SYSTEM_PROMPT = """
You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. You should output in a format similar to <think>...</think><answer>...</answer>, where <think> contains the reasoning process and <answer> contains the final answer. Return final answer within \\boxed{}, after taking modulo 1000. 
"""

train_data_list = read_jsonl_file("math_data/V1_filtered/train_data_filtered.jsonl")
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
    r=8,  # LoRA 的秩
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
def tokenize_function(examples):
    # 将 prompt 和 answer 拼接为输入
    prompts = [ SYSTEM_PROMPT + "\n" + example for example in examples["prompt"] ]
    answers = []
    for answer in examples["answer"]:
        try:
            answers.append(int(answer) % 1000)
        except Exception:
            answers.append(0)
    
    answer_texts = [f"<think>{think}</think>\n" \
                  f"<answer>{answer}. So the mod 1000 answer is \\boxed{{{str(answer)}}}</answer>" for think, answer, answer_mod in zip(examples["think"], examples['answer'], answers)]

    input_text = [prompt + " " + answer for prompt, answer in zip(prompts, answer_texts)]
    tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=512)

    # 创建 labels，只保留 answer 部分的 token
    labels = tokenizer(answer_texts, padding="max_length", truncation=True, max_length=512)["input_ids"]

    # 将非 answer 部分的 token 设置为 -100，避免计算损失
    index = 0
    for prompt, answer in zip(prompts, answers):
        labels[index] = ([-100]*len(prompts) + labels[index][len(prompts):512])
        index += 1

    tokenized["labels"] = labels

    print(len(tokenized["labels"]))
    print(len(tokenized["input_ids"]))

    return tokenized

# 预处理数据集
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 过滤掉空样本
def filter_empty_samples(example):
    return len(example["input_ids"]) > 0 and len(example["answer"]) > 0

tokenized_datasets = tokenized_datasets.filter(filter_empty_samples)

# 检查数据集中是否有空的 input_ids 或 labels
for split in ["train", "validation"]:
    for example in tokenized_datasets[split]:
        assert len(example["input_ids"]) > 0, f"Empty input_ids in {split}"
        assert len(example["labels"]) > 0, f"Empty labels in {split}"

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