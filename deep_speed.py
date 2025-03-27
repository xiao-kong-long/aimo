import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Features, Value, Dataset, DatasetDict
from sft import read_jsonl_file
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B"  # 可以换成任意支持的模型
MODEL_NAME = "gpt2"

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

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(local_rank)
model = DDP(model, device_ids=[local_rank])

tokenizer.pad_token = tokenizer.eos_token


# 数据预处理函数
def tokenize_function(examples):
    # 对 prompt 进行编码
    tokenized = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=128)
    # 设置 labels，与 input_ids 相同
    tokenized["labels"] = tokenized["input_ids"].copy()
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

for batch in tokenized_datasets["train"]:
    print(batch)
    break

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    deepspeed="aimo/ds_config.json",  # 指定 DeepSpeed 配置文件
    fp16=True,  # 启用 FP16 精度
    report_to="none",  # 不使用 WandB 等外部报告
    remove_unused_columns=False
)

# # 手动训练循环
# for epoch in range(training_args.num_train_epochs):
#     for batch in tokenized_datasets["train"]:
#         inputs = {key: torch.tensor(val).to("cuda") for key, val in batch.items()}
#         outputs = model(**inputs)
#         loss = outputs.loss
#         loss.backward()
#         # 优化器更新逻辑

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
final_checkpoint_path = "finetuned_model/deepseek-sft/final"
os.makedirs(final_checkpoint_path, exist_ok=True)
model.save_pretrained(final_checkpoint_path)
tokenizer.save_pretrained(final_checkpoint_path)
