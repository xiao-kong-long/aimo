# Import required libraries
import torch           # Main PyTorch library
from peft import get_peft_model, LoraConfig, TaskType  # For efficient finetuning using QLoRA
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # Hugging Face model components
from torch.optim import AdamW    # Optimizer for training
from tqdm import tqdm   # Progress bar utilities
import os

from aimo.sft_advanced import set_seed, get_hyperparameters, download_and_prepare_data, calculate_accuracy, test_model

# Main training script
if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seed(42)

    # Configure basic training parameters
    train_url = "data/tagmynews3/train.jsonl"
    test_url = "data/tagmynews3/dev.jsonl"
    model_name = "/home/weidu/huqifeng/llm/InternLM2/autodl-tmp/Shanghai_AI_Laboratory/internlm2-chat-7b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,       # 开启 8-bit 量化
        llm_int8_threshold=6.0   # 默认即可，也可根据需要调整
    )

    # Configure QLoRA parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Set task type for causal language modeling
        inference_mode=False,          # Enable training mode
        r=4,                           # Rank of LoRA update matrices
        lora_alpha=16,                 # LoRA scaling factor
        target_modules=["wqkv", "wo"],  # Specify target modules for LoRA
    )

    # Load model and apply QLoRA configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",  # Automatically map model to available devices
        load_in_4bit=True   # Enable 4-bit quantization
    )

    for name, param in model.named_parameters():
        print(name, param.shape)


    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # Get hyperparameters and prepare data
    num_epochs, batch_size, learning_rate = get_hyperparameters()
    train_loader, test_loader = download_and_prepare_data(train_url, test_url, tokenizer, batch_size)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        record_batch_step = 1
        for input_ids, attention_mask, labels, _, _ in progress_bar:
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            if num_batches % record_batch_step == 0:
                test_acc = calculate_accuracy(model, tokenizer, test_loader)
                print(f"Epoch {epoch+1} Batch {num_batches} - Loss: {loss.item():.4f}, Test accuracy: {test_acc:.4f}")
                checkpoint_path = f"finetuned_model/gpt2_generation_qlora/{epoch+1}_{num_batches}"
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

            progress_bar.set_postfix({"Loss": total_loss / num_batches})

        # Calculate and display epoch metrics
        avg_loss = total_loss / num_batches
        test_acc = calculate_accuracy(model, tokenizer, test_loader)
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Calculate final model performance
    train_acc = calculate_accuracy(model, tokenizer, train_loader)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the trained model and tokenizer
    final_checkpoint_path = "finetuned_model/gpt2_generation_qlora/final"
    os.makedirs(final_checkpoint_path, exist_ok=True)
    model.save_pretrained(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    # Test model with a sample input
    test_input = "I'm so happy to be able to finetune an LLM!"
    test_model(final_checkpoint_path, test_input)