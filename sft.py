# Import required libraries
import json             # For parsing JSON data
import random          # For setting seeds and shuffling data
import gzip            # For decompressing dataset
import requests        # For downloading dataset from URL
import torch           # Main PyTorch library
from torch.utils.data import Dataset, DataLoader  # For dataset handling
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face model components
from torch.optim import AdamW    # Optimizer for training
from tqdm import tqdm   # Progress bar utilities
import re              # For text normalization
import os
import matplotlib.pyplot as plt
import logging
from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

SYSTEM_PROMPT = """
You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. You should output in a format similar to <think>...</think><answer>...</answer>, where <think> contains the reasoning process and <answer> contains the final answer. Return final answer within \\boxed{{}}, after taking modulo 1000.

Here we give some examples:
    question: What is the greatest value that the sum $S_\{n\}$ of the first $n$ terms of an arithmetic progression can take, given that the sum $S_{3}=327$ and the sum $S_{57}=57$?, 
    answer: <think> Solution:\n\nIf $a-$ is the first term and $d-$ is the common difference of the arithmetic progression,\n\n$\\left\\{\\begin{array}{l}\\frac{a+a+2 d}{2} \\cdot 3=327, \\\\ \\frac{a+a+56 d}{2} \\cdot 57=57\\end{array} \\Leftrightarrow\\left\\{\\begin{array}{l}a+d=109, \\\\ a+28 d=1\\end{array} \\Rightarrow 27 d=-108 ; d=-4, a=113\\right.\\right.$.\n\nThe sum of the first $n$ terms of the arithmetic progression $S_{n}$ reaches its maximum value if $a_{n}>0$, and $a_{n+1} \\leq 0$. Since $a_{n}=a+d(n-1)$, from the inequality $113-4(n-1)>0$ we find $n=[117 / 4]=29$. Then $\\max S_{n}=S_{29}=0.5 \\cdot(113+113-4 \\cdot 28) \\cdot 29=1653 . \\quad$ </think>
    <answer>1653, so the answer mod 1000 is \\boxed{653} </answer>

Question:
"""

def set_seed(seed):
    """
    Sets random seeds for reproducibility across different libraries.

    Args:
        seed (int): Seed value for random number generation
    """
    # Set Python's built-in random seed
    random.seed(seed)
    # Set PyTorch's CPU random seed
    torch.manual_seed(seed)
    # Set seed for all available GPUs
    torch.cuda.manual_seed_all(seed)
    # Request cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN's auto-tuner for consistent behavior
    torch.backends.cudnn.benchmark = False

def encode_text(tokenizer, text, return_tensor=False):
    """
    Encodes text using the provided tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer
        text (str): Text to encode
        return_tensor (bool): Whether to return PyTorch tensor

    Returns:
        List or tensor of token IDs
    """
    # If tensor output is requested, encode with PyTorch tensors
    if return_tensor:
        return tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        )
    # Otherwise return list of token IDs
    else:
        return tokenizer.encode(text, add_special_tokens=False)

def decode_text(tokenizer, token_ids):
    """
    Decodes token IDs back to text.

    Args:
        tokenizer: Hugging Face tokenizer
        token_ids: List or tensor of token IDs

    Returns:
        str: Decoded text
    """
    # Convert token IDs back to text, skipping special tokens
    return tokenizer.decode(token_ids, skip_special_tokens=True)

class PromptCompletionDataset(Dataset):
    """
    PyTorch Dataset for prompt-completion pairs.
    Handles the conversion of text data into model-ready format.

    Args:
        data (list): List of dictionaries containing prompts and completions
        tokenizer: Hugging Face tokenizer
    """
    def __init__(self, data, tokenizer):
        # Store the raw data and tokenizer for later use
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the total number of examples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        Args:
            idx (int): Index of the example to fetch

        Returns:
            dict: Contains input_ids, labels, prompt, and expected completion
        """
        # Get the specific example from our dataset
        item = self.data[idx]
        prompt = item["prompt"]
        completion = item["completion"]

        # Convert text to token IDs for both prompt and completion
        encoded_prompt = encode_text(self.tokenizer, prompt)
        encoded_completion = encode_text(self.tokenizer, completion)
        # Get the end-of-sequence token ID
        eos_token = self.tokenizer.eos_token_id

        # Combine prompt and completion tokens with EOS token
        input_ids = encoded_prompt + encoded_completion + [eos_token]
        # Create labels: -100 for prompt (ignored in loss), completion tokens for learning
        labels = [-100] * len(encoded_prompt) + encoded_completion + [eos_token]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt": prompt,
            "expected_completion": completion
        }

def collate_fn(batch, tokenizer):
    """
    Collates batch of examples into training-ready format.
    Handles padding and conversion to tensors.

    Args:
        batch: List of examples from Dataset

    Returns:
        tuple: (input_ids, attention_mask, labels, prompts, expected_completions)
    """
    # Find the longest sequence in the batch for padding
    max_length = max(len(item["input_ids"]) for item in batch)

    # Pad input sequences to max_length with pad token
    input_ids = [
        item["input_ids"] +
        [tokenizer.pad_token_id] * (max_length - len(item["input_ids"]))
        for item in batch
    ]

    # Pad label sequences with -100 (ignored in loss calculation)
    labels = [
        item["labels"] +
        [-100] * (max_length - len(item["labels"]))
        for item in batch
    ]

    # Create attention masks: 1 for real tokens, 0 for padding
    attention_mask = [
        [1] * len(item["input_ids"]) +
        [0] * (max_length - len(item["input_ids"]))
        for item in batch
    ]

    # Keep original prompts and completions for evaluation
    prompts = [item["prompt"] for item in batch]
    expected_completions = [item["expected_completion"] for item in batch]

    # Convert everything to PyTorch tensors except text
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(labels),
        prompts,
        expected_completions
    )

def normalize_text(text):
    """
    Normalizes text for consistent comparison.

    Args:
        text (str): Input text

    Returns:
        str: Normalized text
    """
    # Remove leading/trailing whitespace and convert to lowercase
    text = text.strip().lower()
    # Replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""

#################################### multi processing inference ##########################################

def calculate_accuracy_batch(model, tokenizer, loader):
    """
    Calculates prediction accuracy on a dataset.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        loader: DataLoader containing evaluation examples

    Returns:
        float: Accuracy score
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    correct = 0
    total = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for input_ids, attention_mask, labels, prompts, expected_completions in loader:
            # Batch encode prompts
            device = next(model.parameters()).device
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            # Generate predictions for the entire batch
            if isinstance(model, torch.nn.DataParallel):
                output_ids = model.module.generate(
                    input_ids=input_ids["input_ids"],
                    attention_mask=input_ids["attention_mask"],
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                    do_sample=False
                )
            else:
                print('len of input_ids:', len(input_ids["input_ids"]))
                output_ids = model.generate(
                    input_ids=input_ids["input_ids"],
                    attention_mask=input_ids["attention_mask"],
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                    do_sample=False
                )

            # Decode the generated outputs
            generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # Compare results
            for generated_text, expected_completion in zip(generated_texts, expected_completions):
                generated_result = extract_boxed_text(generated_text)
                expected_result = extract_boxed_text(expected_completion)
                
                DEBUG = False  # 设置为 True 以启用调试打印

                if DEBUG:
                    print("=" * 50)
                    print("generated_text:", generated_text)
                    print("generated_result:", generated_result)
                    print("expected_result:", expected_result)
                
                if generated_result == expected_result:
                    correct += 1
                total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    model.train()
    return accuracy

##########################################################################################################




def calculate_accuracy(model, tokenizer, loader):
    """
    Calculates prediction accuracy on a dataset.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        loader: DataLoader containing evaluation examples

    Returns:
        float: Accuracy score
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    # Initialize counters for accuracy calculation
    correct = 0
    total = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate through batches
        for input_ids, attention_mask, labels, prompts, expected_completions in loader:
            # Process each example in the batch
            for prompt, expected_completion in zip(prompts, expected_completions):
                # Generate model's prediction for this prompt
                generated_text = generate_text(model, tokenizer, prompt)
                # Compare normalized versions of prediction and expected completion
                generated_result = extract_boxed_text(generated_text)
                expected_result = extract_boxed_text(expected_completion)
                if generated_result == expected_result:
                    correct += 1
                # if normalize_text(generated_text) == normalize_text(expected_completion):
                    # correct += 1
                total += 1

    # Calculate accuracy, handling empty dataset case
    accuracy = correct / total if total > 0 else 0
    # Reset model to training mode
    model.train()
    return accuracy

def generate_text(model, tokenizer, prompt, max_new_tokens=2048):
    """
    Generates text completion for a given prompt.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        prompt (str): Input prompt
        max_new_tokens (int): Maximum number of tokens to generate

    Returns:
        str: Generated completion
    """
    # # Encode prompt and move to model's device
    # input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 根据模型类型获取设备
    if isinstance(model, torch.nn.DataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate completion using model
    if isinstance(model, torch.nn.DataParallel):
        output_ids = model.module.generate(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,        # Use KV cache for faster generation
            num_beams=1,           # Use greedy decoding
            do_sample=False,       # Don't use sampling
        )[0]
    else:
        output_ids = model.generate(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,        # Use KV cache for faster generation
            num_beams=1,           # Use greedy decoding
            do_sample=False,       # Don't use sampling
        )[0]

    # Extract and decode only the generated part (excluding prompt)
    generated_text = decode_text(tokenizer, output_ids[input_ids["input_ids"].shape[1]:])
    return generated_text.strip()

def test_model(model_path, test_input):
    """
    Tests a saved model on a single input.

    Args:
        model_path (str): Path to saved model
        test_input (str): Text to classify
    """
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load saved model and move to appropriate device
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    if torch.cuda.device_count() > 1:
        print('--------------------------------HAHAHA--------------------------------')
        model = torch.nn.DataParallel(model)
        # model.to(device)
    

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure model has proper padding token configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create prompt and generate prediction
    prompt = build_prompt(test_input)
    generated_text = generate_text(model, tokenizer, prompt)

    # Display results
    print(f"Input: {test_input}")
    print(f"Generated emotion: {generated_text}")

def read_jsonl_file(file_path):
    """
    Reads a JSON Lines (.jsonl) file and returns a list of JSON objects.

    Args:
        file_path (str): Path to the JSON Lines file

    Returns:
        list: List of JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_and_split_dataset_from_different_source(train_url, test_url):
    train_dataset = read_jsonl_file(train_url)
    test_dataset = read_jsonl_file(test_url)
    return train_dataset, test_dataset

def load_dataset_from_single_source(url):
    dataset = read_jsonl_file(url)
    return dataset

def download_and_prepare_data(url, tokenizer, batch_size, test_ratio=0.1):
    """
    Downloads and prepares dataset for training.

    Args:
        data_url (str): URL of the dataset
        tokenizer: Tokenizer for text processing
        batch_size (int): Batch size for DataLoader
        test_ratio (float): Proportion of data for testing

    Returns:
        tuple: (train_loader, test_loader)
    """

    # Load and split dataset
    train_json = load_dataset_from_single_source(url)

    # # Download compressed dataset
    # response = requests.get(data_url)
    # # Decompress and decode the content
    # content = gzip.decompress(response.content).decode()

    # Parse each line as JSON and format into prompt-completion pairs
    # dataset = []
    # for entry in map(json.loads, content.splitlines()):
    #     dataset.append({
    #         "prompt": build_prompt(entry['text']),
    #         "completion": entry["label"].strip()
    #     })

    train_data = []
    for entry in train_json:
        # train_data.append({
        #     "prompt": build_prompt(entry['text']),
        #     "completion": entry["label"].strip()
        # })
        answer = entry["answer"].strip()

        try:
            
            answer = int(answer) % 1000
            answer = str(answer)
        except:
            # logging.error(f"Error in answer: {answer}")
            
            # print("====================================")
            # print(answer.isdigit())
            # print(type(answer))
            # print('answer:', answer)
            # print(entry["prompt"])
            # print(type(entry["think"]))
            # print(type(entry["answer"]))
            continue

        if not isinstance(entry['prompt'], str):
            continue
        
        # print("="*50)
        # print("content:", entry)
        train_data.append({
            "prompt": build_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": entry["prompt"]}
            ]),
            "completion": f"<think>{entry['think']}</think>\n" \
                  f"<answer>{entry['answer']}. So the mod 1000 answer is \\boxed{{{str(answer)}}}</answer>"
        })


    # # Randomly shuffle dataset for better split
    # random.shuffle(dataset)
    # # Calculate split index based on test ratio
    # split_index = int(len(dataset) * (1 - test_ratio))
    # # Split into train and test sets
    # train_data = dataset[:split_index]
    # test_data = dataset[split_index:]

    # Create dataset objects
    train_dataset = PromptCompletionDataset(train_data, tokenizer)

    # Create data loaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,         # Shuffle training data
        collate_fn=lambda batch: collate_fn(batch, tokenizer)  # Custom collation for padding
    )

    return train_loader

def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    # print('messages:', messages)
    return "\n".join([msg["content"].strip() for msg in messages])


def get_hyperparameters():
    """
    Returns training hyperparameters.

    Returns:
        tuple: (num_epochs, batch_size, learning_rate)
    """
    # Training for 2 epochs as a balance of learning and efficiency
    num_epochs = 2
    # Batch size of 16 works well with most GPU memory sizes
    batch_size = 16
    # Standard learning rate for fine-tuning transformers
    learning_rate = 5e-5

    return num_epochs, batch_size, learning_rate

# 创建实时绘图函数
def update_plot():
    plt.clf()  # 清除当前图像
    plt.subplot(2, 1, 1)  # 创建第一个子图：损失曲线
    plt.plot(batches, losses, label="Loss", color="blue")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)  # 创建第二个子图：测试准确率曲线
    plt.plot(batches, accuracies, label="Test Accuracy", color="green")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()  # 自动调整子图布局
    plt.pause(0.01)  # 暂停以更新图像

# Main training script
if __name__ == "__main__":

    # plot the training loss
    # 初始化绘图数据
    losses = []
    accuracies = []
    batches = []

    # 初始化 GradScaler
    scaler = GradScaler()

    torch.backends.cuda.enable_flash_sdp(True)  # 启用FlashAttention（如果可用）
    torch.backends.cuda.enable_mem_efficient_sdp(True)



    # Set random seeds for reproducibility
    set_seed(42)

    # Configure basic training parameters
    # Configure training parameters
    # model_name = "DeepSeek-R1-Distill-Qwen-7B"
    model_name = "gpt2"

    is_soretd = False
    if not is_soretd:
        train_url = "math_data/V1_filtered/train_data_filtered.jsonl"
        test_large_url = "math_data/V1_filtered/test_large_data_filtered.jsonl"
        test_small_url = "math_data/V1_filtered/test_small_data_filtered.jsonl"
    else:
        train_url = "math_data/V2_sorted/train_data_filtered_sorted.jsonl"
        test_large_url = "math_data/V2_sorted/test_large_data_filtered_sorted.jsonl"
        test_small_url = "math_data/V2_sorted/test_small_data_filtered_sorted.jsonl"

    train_url = "math_data/train.jsonl"
    test_large_url = "math_data/train.jsonl"
    test_small_url = "math_data/train.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained model and move to appropriate device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用 float16 数据类型
        device_map=None,
        trust_remote_code=True  # 信任远程模型代码
    ).to(device)

    if torch.cuda.device_count() > 1:
        print('--------------------------------HAHAHA--------------------------------')
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        print(next(model.parameters()).device)  # 应输出 "cuda:0"
        print(torch.cuda.memory_allocated(0))  # 查看 GPU 0 显存占用
        print(torch.cuda.memory_allocated(1))  # 查看 GPU 1 显存占用
    
    # model.to(device)

    # Initialize tokenizer and configure padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained model and move to appropriate device
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    model = torch.nn.DataParallel(model, device_ids=[0,1])  # 使用 DataParallel 包装模型


    # Get training hyperparameters
    num_epochs, batch_size, learning_rate = get_hyperparameters()

    # Load and prepare training data
    train_loader = download_and_prepare_data(train_url, tokenizer, batch_size)
    test_large_loader = download_and_prepare_data(test_large_url, tokenizer, batch_size)
    test_small_loader = download_and_prepare_data(test_small_url, tokenizer, batch_size)

    # Initialize optimizer with learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Initialize epoch metrics
        total_loss = 0
        num_batches = 0
        # Create progress bar for this epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Process each batch
        record_batch_step = 1
        for input_ids, attention_mask, labels, _, _ in progress_bar:
            # Move batch data to appropriate device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass with loss calculation
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss.mean()

            # Backward pass and optimization with GradScaler
            scaler.scale(loss).backward()  # 缩放梯度
            scaler.step(optimizer)         # 更新参数
            scaler.update()                # 更新缩放因子
            optimizer.zero_grad()          # 清空梯度

            # Update progress metrics
            total_loss += loss.item()
            num_batches += 1

            if num_batches % record_batch_step == 0:
                test_small_acc = calculate_accuracy(model, tokenizer, test_small_loader)
                print(f"Epoch {epoch+1} Batch {num_batches} - Loss: {loss.item():.4f}, Test Small accuracy: {test_small_acc:.4f}")
                checkpoint_path = f"finetuned_model/deepseek-sft/{epoch+1}_{num_batches}"
                os.makedirs(checkpoint_path, exist_ok=True)
                if isinstance(model, torch.nn.DataParallel):
                    model.module.save_pretrained(checkpoint_path)
                else:
                    model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

                # 保存损失和准确率数据
                losses.append(loss.item())
                accuracies.append(test_small_acc)
                batches.append(num_batches)

                # 更新实时绘图
                update_plot()

            # Update progress bar with current loss
            progress_bar.set_postfix({"Loss": total_loss / num_batches})

        # Calculate and display epoch metrics
        avg_loss = total_loss / num_batches
        test_small_acc = calculate_accuracy(model, tokenizer, test_small_loader)
        test_large_acc = calculate_accuracy(model, tokenizer, test_large_loader)
        print("="*50)
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}, Test Small accuracy: {test_small_acc:.4f}, Test Large accuracy: {test_large_acc:.4f}")

    # Calculate final model performance
    train_acc = calculate_accuracy(model, tokenizer, train_loader)
    print(f"Training accuracy: {train_acc:.4f}")

    # Save the trained model and tokenizer
    final_checkpoint_path = "finetuned_model/deepseek-sft/final"
    os.makedirs(final_checkpoint_path, exist_ok=True)
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(checkpoint_path)
    else:
        model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    # 显示最终图像
    plt.show()